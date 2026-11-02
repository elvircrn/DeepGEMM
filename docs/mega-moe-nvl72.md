# Mega MoE across the OS instances of one NVLink fabric (NVL72)

Mega MoE runs EP dispatch, the expert GEMMs and EP combine inside one kernel
that reads and writes peer GPU memory directly over NVLink. The kernels address
peers through plain pointers (`layout::SymBuffer` maps a local pointer plus a
per-rank offset), so the only requirement beyond the fabric itself is that
every rank can map every other rank's buffer into its own address space.

## The problem on a partitioned NVL72

A GB200 NVL72 rack is one NVLink fabric, but it is partitioned into multiple
OS instances (down to one instance per Grace CPU). `torch._symmetric_memory`
exchanges `cudaIpcMemHandle`s, which are only valid inside one OS instance, so
with the stock buffer setup Mega MoE cannot reach peer GPUs on the other
instances even though the fabric connects them: an EP group spanning 4
instances x 8 GPUs fails during buffer setup, not in the kernel.

## Fabric handles

`SymmBuffer` (and `NVFP4SymmBuffer`) can instead set the buffer up through
CUDA fabric handles:

1. Each rank allocates its buffer with `cuMemCreate`, requesting
   `CU_MEM_HANDLE_TYPE_FABRIC`, and maps it into its own address space.
2. `cuMemExportToShareableHandle` returns a 64-byte `CUmemFabricHandle` of
   plain data. Unlike an IPC handle it is not a file descriptor, so it travels
   to the other ranks over any process group (an `all_gather` of eight int64
   words).
3. Every rank imports each peer's descriptor with
   `cuMemImportFromShareableHandle` + `cuMemMap` and grants its own GPU
   read/write access.

The result is the same `buffer_ptrs` list the kernels have always consumed,
except the mappings span every OS instance of the fabric. Nothing in the
kernels changes, and the same code path works inside a single instance.

Selection is controlled with `DG_SYMM_BUFFER_BACKEND`:

| Value | Behavior |
|---|---|
| `auto` (default) | try `torch._symmetric_memory` first; fall back to fabric handles when it fails, which is exactly the cross-OS-instance case |
| `fabric` | force fabric handles; fail loudly if unsupported |
| `symm_mem` | force the stock `torch._symmetric_memory` path |

The fallback is symmetric: `torch._symmetric_memory` exchanges its handles
through collectives and then fails the same way on every rank, so all ranks
land on fabric handles together. On a partitioned NVL72 you can set
`DG_SYMM_BUFFER_BACKEND=fabric` to skip the doomed `torch._symmetric_memory`
attempt; on single-instance clusters the default changes nothing, and setting
`fabric` there is a way to exercise this path on hardware you already have
(the mappings have the same semantics as the IPC ones).

## Prerequisites

The same ones NCCL's MNNVL transport has:

- CUDA 12.1 or newer (headers and driver).
- The NVLink fabric up on every OS instance: `nvidia-smi -q -d SUPPORTED`
  must show fabric state `COMPLETED` on every participating GPU.
- The IMEX service configured on every instance:
  `/dev/nvidia-caps-imex-channels` present, `nvidia-imex-ctl -N` healthy.
  A missing IMEX configuration is the most common reason fabric handle
  export/import fails on an otherwise healthy fabric.
- Identical GPUs across the EP group (the descriptors are only meaningful
  between matching devices on one fabric).

A zero-code smoke test before anything else: run an `nccl-tests` collective
across the instances with `NCCL_DEBUG=INFO NCCL_MNNVL_ENABLE=1` and look for
the `MNNVL 1 cliqueId ... nvlDomainSize` log line. If NCCL reports MNNVL
there, every API used here is live on that fabric.

## Testing

Buffer visibility only (reads and writes of peer mappings, no MoE math); the
test forces the fabric path, so it also runs on a single instance:

```bash
# One instance
torchrun --nproc_per_node 8 tests/test_fabric_symm_buffer.py

# Across the instances of a partitioned NVL72
MASTER_ADDR=<head ip> MASTER_PORT=29500 \
    torchrun --nnodes 4 --nproc_per_node <gpus per node> tests/test_fabric_symm_buffer.py
```

Full kernel numerics (dispatch TMA pulls and combine writes go through the
same peer mappings, so this is also the TMA-over-fabric check):

```bash
DG_SYMM_BUFFER_BACKEND=fabric torchrun --nproc_per_node 8 tests/test_mega_moe.py
```

## Serving with vLLM (DP attention + EP)

On a 4-instance NVL72 partition with 8 GPUs per instance, run one vLLM
process per GPU with the EP group spanning all 32 GPUs (e.g. `--dp 32` with
DSV4.1-style DP attention, MoE EP across the whole group). On Blackwell use
the SM100 Mega MoE kernels; the buffer setup code is architecture
independent. The EP group must still fit inside one NVLink fabric: crossing
InfiniBand between racks is a different problem (hierarchical dispatch, or a
fabric-capable transport for the cross-rack hop) and is not addressed by
fabric handles.

## Notes

- Fabric allocations are granularity-aligned (2 MB typical), so the buffer can
  be slightly larger than requested. The `base` reuse path already tolerates a
  larger buffer.
- `destroy` is collective: every rank calls it after a `group.barrier()`, and
  only the instance that allocated frees (buffers reused through `base`
  share the mapping).
- The kernels do not use NVSwitch multicast yet; fabric handles only provide
  point-to-point peer mappings, the same semantics the IPC path provides
  inside one instance.
