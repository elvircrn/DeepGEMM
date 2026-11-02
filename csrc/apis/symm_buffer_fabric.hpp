#pragma once

// Fabric-handle symmetric buffer for Mega MoE.
//
// `torch.distributed._symmetric_memory` exchanges `cudaIpcMemHandle`s, which
// are only valid inside one OS instance. A GB200 NVL72 rack is one NVLink
// fabric split across multiple OS instances, so with the stock setup the Mega
// MoE kernels cannot reach peer GPUs on the other instances.
//
// `CU_MEM_HANDLE_TYPE_FABRIC` lifts that limit: `cuMemExportToShareableHandle`
// yields a 64-byte `CUmemFabricHandle` of plain data (no file descriptor), so
// the descriptor travels to the other ranks over any process group. Importing
// it with `cuMemImportFromShareableHandle` + `cuMemMap` maps the peer's
// physical allocation into the local address space, which is exactly the
// `buffer_ptrs` layout `layout::SymBuffer` already consumes.
//
// Prerequisites, the same ones NCCL's MNNVL transport has: the NVLink fabric
// must be up on every OS instance (NVML fabric state COMPLETED) and the IMEX
// service must be configured (`nvidia-imex-ctl -N`,
// `/dev/nvidia-caps-imex-channels`). Unlike the IPC path, buffers set up this
// way span every OS instance of one fabric, e.g. a partitioned NVL72.
//
// Ownership: the tensor returned by `alloc` owns the rank's whole mapping set
// (its own allocation plus one import per peer) through its storage deleter,
// so views and buffers reusing it through `base` keep the mappings alive and
// a dropped buffer frees them, the same refcounted lifetime
// `torch._symmetric_memory` provides. VMM keeps a physical allocation alive
// while any rank still maps it, so ranks can tear down independently.

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>

#include <deep_jit/backend/cuda/driver.hpp>

#include "../utils/exception.hpp"
#include "../utils/torch_compat.hpp"

// Fabric handles exist since CUDA 12.1, matching NCCL's guard
// (`CUDART_VERSION >= 12010` around its `NVLS_HANDLE_SIZE`).
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12010
#define DEEP_GEMM_FABRIC_SYMM_BUFFER 1
#else
#define DEEP_GEMM_FABRIC_SYMM_BUFFER 0
#endif

namespace deep_gemm::fabric_symm_buffer {

#if DEEP_GEMM_FABRIC_SYMM_BUFFER

static constexpr size_t kFabricHandleNumWords = sizeof(CUmemFabricHandle) / sizeof(int64_t);

struct Mapping {
    size_t num_bytes;
    CUmemGenericAllocationHandle handle;
};

// Outstanding mappings and the groups they belong to, guarded by `g_mutex`.
// A group is the set of mappings one rank holds for one symmetric buffer: its
// own allocation plus one import per peer, keyed by the local allocation
// address (the `buffer_ptrs[rank]` entry). A group dies as a unit, through
// `fabric_symm_buffer_free` or the storage deleter of the buffer tensor.
static std::mutex g_mutex;
static std::map<CUdeviceptr, Mapping> g_mappings;
static std::map<CUdeviceptr, std::vector<CUdeviceptr>> g_groups;

static int current_device() {
    int device;
    DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&device));
    return device;
}

static torch::stable::Device current_cuda_device() {
    return torch::stable::Device(
        torch::headeronly::kCUDA, torch::stable::accelerator::getCurrentDeviceIndex());
}

static size_t granularity() {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = current_device();
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t value = 0;
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemGetAllocationGranularity(
        &value, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    DG_HOST_ASSERT(value > 0);
    return value;
}

static size_t align_up(const size_t& num_bytes, const size_t& alignment) {
    return (num_bytes + alignment - 1) / alignment * alignment;
}

// Map `handle` at a fresh address, readable and writable by the current device.
static CUdeviceptr map_allocation(const CUmemGenericAllocationHandle& handle, const size_t& num_bytes) {
    const size_t alignment = granularity();
    CUdeviceptr address = 0;
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemAddressReserve(&address, num_bytes, alignment, 0, 0));
    bool is_mapped = false;
    try {
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemMap(address, num_bytes, 0, handle, 0));
        is_mapped = true;
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = current_device();
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemSetAccess(address, num_bytes, &access, 1));
    } catch (...) {
        if (is_mapped) {
            DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemUnmap(address, num_bytes));
        }
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemAddressFree(address, num_bytes));
        throw;
    }
    return address;
}

static void track_mapping(const CUdeviceptr& address, const Mapping& mapping) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_mappings.emplace(address, mapping);
}

// Register `address` (the local allocation or an import) as part of the group
// keyed by `group_key`, the owner's local allocation address
static void join_group(const CUdeviceptr& group_key, const CUdeviceptr& address) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_groups[group_key].push_back(address);
}

static void forget_group(const CUdeviceptr& group_key) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_groups.erase(group_key);
}

// Unmap and release every mapping of one group. Idempotent: a group that is
// already gone is a no-op, so the storage deleter and an explicit
// `fabric_symm_buffer_free` can both run for the same buffer.
static void free_group(const CUdeviceptr& group_key) {
    std::vector<CUdeviceptr> addresses;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto group = g_groups.find(group_key);
        if (group == g_groups.end()) {
            return;
        }
        addresses = std::move(group->second);
        g_groups.erase(group);
    }
    // In-flight kernels may still touch the mappings; wait before unmapping
    DG_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    for (const auto& address : addresses) {
        Mapping mapping{};
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            const auto it = g_mappings.find(address);
            if (it == g_mappings.end()) {
                continue;
            }
            mapping = it->second;
            g_mappings.erase(it);
        }
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemUnmap(address, mapping.num_bytes));
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemRelease(mapping.handle));
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemAddressFree(address, mapping.num_bytes));
    }
}

static bool is_supported() {
    CUdevice device = 0;
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuDeviceGet(&device, current_device()));
    int value = 0;
    // The attribute is absent on older drivers; treat any error as unsupported, like NCCL does
    const CUresult result = deep_jit::cuda::driver::lazy_cuDeviceGetAttribute(
        &value, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
    return result == CUDA_SUCCESS and value != 0;
}

static std::tuple<torch::stable::Tensor, std::vector<int64_t>> alloc(const int64_t& num_bytes) {
    DG_HOST_ASSERT(num_bytes > 0);
    const auto aligned = align_up(num_bytes, granularity());

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = current_device();
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemGenericAllocationHandle handle = 0;
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemCreate(&handle, aligned, &prop, 0));

    CUdeviceptr address = 0;
    bool owned_by_tensor = false;
    try {
        address = map_allocation(handle, aligned);
        CUmemFabricHandle fabric{};
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemExportToShareableHandle(
            &fabric, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
        track_mapping(address, Mapping(aligned, handle));
        join_group(address, address);
        std::vector<int64_t> words(kFabricHandleNumWords);
        std::memcpy(words.data(), &fabric, sizeof(fabric));

        // Int8 view over the mapping. Its storage deleter owns the whole group
        // (this mapping plus the peer imports registered under it), so views
        // and buffers reusing this one through `base` keep the mappings alive
        // and a dropped buffer frees them instead of leaking. The deleter runs
        // wherever the storage dies, possibly during interpreter teardown, so
        // it must not throw. Everything fallible is done before the tensor
        // exists; once it does, the deleter is the only cleanup path.
        const CUdeviceptr group_key = address;
        auto buffer = torch::stable::from_blob(
            reinterpret_cast<void*>(address),
            std::vector<int64_t>{static_cast<int64_t>(aligned)}, std::vector<int64_t>{1},
            current_cuda_device(), torch::headeronly::ScalarType::Byte,
            [group_key](void*) {
                try {
                    free_group(group_key);
                } catch (...) {
                }
            });
        owned_by_tensor = true;
        return {buffer, std::move(words)};
    } catch (...) {
        // If the tensor was created, its destruction already ran the deleter
        // and freed the group; freeing again would release handle and address
        // twice
        if (not owned_by_tensor) {
            forget_group(address);
            if (address != 0) {
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    g_mappings.erase(address);
                }
                DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemUnmap(address, aligned));
                DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemAddressFree(address, aligned));
            }
            DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemRelease(handle));
        }
        throw;
    }
}

// Import a peer's descriptor and register the mapping in the group keyed by
// `group_key`, the local allocation address of the buffer it belongs to
static int64_t import(const std::vector<int64_t>& words, const int64_t& num_bytes, const int64_t& group_key) {
    DG_HOST_ASSERT(num_bytes > 0);
    DG_HOST_ASSERT(words.size() == kFabricHandleNumWords);
    // The descriptors are only meaningful between identical GPUs on one fabric,
    // so the local alignment of `num_bytes` matches the owner's
    const auto aligned = align_up(num_bytes, granularity());

    CUmemFabricHandle fabric{};
    std::memcpy(&fabric, words.data(), sizeof(fabric));
    CUmemGenericAllocationHandle handle = 0;
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemImportFromShareableHandle(
        &handle, &fabric, CU_MEM_HANDLE_TYPE_FABRIC));
    CUdeviceptr address = 0;
    try {
        address = map_allocation(handle, aligned);
        track_mapping(address, Mapping(aligned, handle));
        join_group(static_cast<CUdeviceptr>(group_key), address);
    } catch (...) {
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemRelease(handle));
        throw;
    }
    return static_cast<int64_t>(address);
}

// Frees the group `data_ptr` belongs to: its own allocation or an imported
// peer mapping. The Python failure path calls this with the local allocation
// address; normal teardown goes through the tensor's storage deleter.
static void free_buffer(const int64_t& data_ptr) {
    const auto address = static_cast<CUdeviceptr>(data_ptr);
    CUdeviceptr group_key = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_groups.count(address) != 0) {
            group_key = address;
        } else {
            for (const auto& [key, members] : g_groups) {
                if (std::find(members.begin(), members.end(), address) != members.end()) {
                    group_key = key;
                    break;
                }
            }
        }
    }
    DG_HOST_ASSERT(group_key != 0);  // Unknown pointer: nothing of ours to free
    free_group(group_key);
}

// Non-owning int8 tensor over a mapping created by `alloc`/`import`, for tests and debugging
static torch::stable::Tensor view_tensor(const int64_t& data_ptr, const int64_t& num_bytes) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_mappings.find(static_cast<CUdeviceptr>(data_ptr));
        DG_HOST_ASSERT(it != g_mappings.end() and num_bytes <= static_cast<int64_t>(it->second.num_bytes));
    }
    return torch_compat::from_blob(
        reinterpret_cast<void*>(static_cast<uintptr_t>(data_ptr)), {num_bytes},
        current_cuda_device(), torch::headeronly::ScalarType::Byte);
}

#else

static bool is_supported() { return false; }
static std::tuple<torch::stable::Tensor, std::vector<int64_t>> alloc(const int64_t&) {
    DG_HOST_UNREACHABLE("Fabric symmetric buffers need CUDA 12.1 or newer");
    return {};
}
static int64_t import(const std::vector<int64_t>&, const int64_t&, const int64_t&) {
    DG_HOST_UNREACHABLE("Fabric symmetric buffers need CUDA 12.1 or newer");
    return 0;
}
static void free_buffer(const int64_t&) {
    DG_HOST_UNREACHABLE("Fabric symmetric buffers need CUDA 12.1 or newer");
}

static torch::stable::Tensor view_tensor(const int64_t&, const int64_t&) {
    DG_HOST_UNREACHABLE("Fabric symmetric buffers need CUDA 12.1 or newer");
    return torch::stable::Tensor();
}

#endif

}  // namespace deep_gemm::fabric_symm_buffer

STABLE_TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("fabric_symm_buffer_supported() -> bool");
    m.def("fabric_symm_buffer_alloc(int num_bytes) -> (Tensor, int[])");
    m.def("fabric_symm_buffer_import(int[] desc, int num_bytes, int group_key) -> int");
    m.def("fabric_symm_buffer_view(int data_ptr, int num_bytes) -> Tensor");
    m.def("fabric_symm_buffer_free(int data_ptr) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CompositeExplicitAutograd, m) {
    // These operators have no tensor argument from which the dispatcher could
    // infer CUDA, even though they allocate or return CUDA tensors.
    using deep_gemm::fabric_symm_buffer::alloc;
    using deep_gemm::fabric_symm_buffer::free_buffer;
    using deep_gemm::fabric_symm_buffer::import;
    using deep_gemm::fabric_symm_buffer::is_supported;
    using deep_gemm::fabric_symm_buffer::view_tensor;
    m.impl("fabric_symm_buffer_supported", TORCH_BOX(&is_supported));
    m.impl("fabric_symm_buffer_alloc", TORCH_BOX(&alloc));
    m.impl("fabric_symm_buffer_import", TORCH_BOX(&import));
    m.impl("fabric_symm_buffer_view", TORCH_BOX(&view_tensor));
    m.impl("fabric_symm_buffer_free", TORCH_BOX(&free_buffer));
}
