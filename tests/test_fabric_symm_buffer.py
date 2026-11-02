"""Fabric-handle symmetric buffer, verified across every rank of the group.

Unlike `torch._symmetric_memory` (CUDA IPC handles, valid inside one OS
instance only), CUDA fabric handles map peer memory across every OS instance
of one NVLink fabric, e.g. a GB200 NVL72 partitioned into multiple nodes.

The test forces the fabric path (DG_SYMM_BUFFER_BACKEND=fabric), so it also
works on a single instance where `auto` would pick `torch._symmetric_memory`.

Single instance (at least 2 GPUs):
    DG_SYMM_BUFFER_BACKEND=fabric torchrun --nproc_per_node 8 tests/test_fabric_symm_buffer.py
    # the env var is the default inside this test; the line above is equivalent to:
    torchrun --nproc_per_node 8 tests/test_fabric_symm_buffer.py

Across the OS instances of a partitioned NVL72 (fabric and IMEX must be up,
see docs/mega-moe-nvl72.md):
    MASTER_ADDR=<head ip> MASTER_PORT=29500 \
        torchrun --nnodes 4 --nproc_per_node <gpus per node> tests/test_fabric_symm_buffer.py

The test allocates a Mega MoE `SymmBuffer` through fabric handles, then checks
that every rank reads and writes every other rank's allocation through its
imported mapping. For the full kernel path (dispatch TMA pulls and combine
writes over the same mappings) run
`DG_SYMM_BUFFER_BACKEND=fabric torchrun --nproc_per_node 8 tests/test_mega_moe.py`.
"""

import os
import sys

os.environ.setdefault('DG_SYMM_BUFFER_BACKEND', 'fabric')

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm import _C


def main() -> int:
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size >= 2, 'Run with at least 2 processes to exercise peer mappings'

    if not _C.fabric_symm_buffer_supported():
        if rank == 0:
            print('SKIP: the device does not report CUDA fabric handle support '
                  '(no NVLink fabric, or the fabric/IMEX service is down)')
        dist.destroy_process_group()
        return 0

    # A small FP8 x FP4 Mega MoE buffer, set up through CUDA fabric handles
    buffer = deep_gemm.SymmBuffer(
        dist.group.WORLD, num_experts=world_size * 2,
        num_max_tokens_per_rank=128, num_topk=8,
        hidden=512, intermediate_hidden=256,
        mma_type='fp8xfp4', activation='swiglu')
    assert buffer.buffer.dtype == torch.int8
    assert buffer.handle.buffer_ptrs[rank] == buffer.buffer.data_ptr()
    if rank == 0:
        print(f'Buffer: {buffer.buffer.nbytes / 1e6:.1f} MB per rank over {world_size} ranks, fabric handles')

    # Every rank fills its own allocation, then all ranks read every peer mapping
    buffer.buffer.fill_(rank + 1)
    dist.barrier()
    for r in range(world_size):
        peer = _C.fabric_symm_buffer_view(buffer.handle.buffer_ptrs[r], buffer.buffer.nbytes)
        assert torch.equal(peer, torch.full_like(peer, r + 1)), \
            f'rank {rank} reads wrong data from rank {r}'
    if rank == 0:
        print(f'Cross-rank reads verified on all {world_size} ranks')

    # Write-through: every rank writes its successor's allocation, owners verify
    successor = (rank + 1) % world_size
    _C.fabric_symm_buffer_view(buffer.handle.buffer_ptrs[successor], buffer.buffer.nbytes).fill_(50 + rank)
    dist.barrier()
    assert bool((buffer.buffer == 50 + (rank - 1) % world_size).all()), \
        f'rank {rank} sees wrong data written by its predecessor'
    if rank == 0:
        print(f'Cross-rank writes verified on all {world_size} ranks')

    # Lifetime: a dropped buffer must free its mappings (the tensor's storage
    # deleter owns them); fabric allocations bypass the caching allocator, so
    # a leak shows up directly in free device memory
    free_before, _ = torch.cuda.mem_get_info()
    for _ in range(4):
        dropped = deep_gemm.SymmBuffer(
            dist.group.WORLD, num_experts=world_size * 2,
            num_max_tokens_per_rank=128, num_topk=8,
            hidden=512, intermediate_hidden=256,
            mma_type='fp8xfp4', activation='swiglu')
        del dropped
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    assert free_after >= free_before - buffer.buffer.nbytes, \
        f'rank {rank}: dropping a SymmBuffer leaked fabric mappings ' \
        f'({free_before - free_after} bytes of device memory lost)'
    if rank == 0:
        print('Dropped buffers free their mappings on all ranks')

    # Lifetime: destroying a base must leave buffers reusing it valid (they
    # share the buffer tensor, whose storage keeps the mappings alive)
    base = deep_gemm.SymmBuffer(
        dist.group.WORLD, num_experts=world_size * 2,
        num_max_tokens_per_rank=128, num_topk=8,
        hidden=512, intermediate_hidden=256,
        mma_type='fp8xfp4', activation='swiglu')
    borrower = deep_gemm.SymmBuffer(
        dist.group.WORLD, num_experts=world_size * 2,
        num_max_tokens_per_rank=128, num_topk=8,
        hidden=512, intermediate_hidden=256,
        mma_type='fp8xfp4', activation='swiglu', base=base)
    base.destroy()
    borrower.buffer.fill_(7)
    assert bool((borrower.buffer == 7).all()), 'base destroy invalidated a reused buffer'
    # Both must go: the base's remaining views keep the storage alive otherwise
    del borrower, base
    if rank == 0:
        print('Buffers reusing a destroyed base stay valid')

    # Teardown
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
