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

// Outstanding mappings, freed by `fabric_symm_buffer_free` (the Python side
// calls it from `SymmBuffer.destroy`, after a group barrier).
static std::mutex g_mutex;
static std::map<CUdeviceptr, Mapping> g_mappings;

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
    try {
        address = map_allocation(handle, aligned);
        CUmemFabricHandle fabric{};
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemExportToShareableHandle(
            &fabric, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
        track_mapping(address, Mapping(aligned, handle));

        // Int8 view over the mapping; ownership stays with `fabric_symm_buffer_free`, not the tensor
        auto buffer = torch_compat::from_blob(
            reinterpret_cast<void*>(address), {static_cast<int64_t>(aligned)},
            current_cuda_device(), torch::headeronly::ScalarType::Byte);
        std::vector<int64_t> words(kFabricHandleNumWords);
        std::memcpy(words.data(), &fabric, sizeof(fabric));
        return {buffer, words};
    } catch (...) {
        if (address != 0) {
            DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemUnmap(address, aligned));
            DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemAddressFree(address, aligned));
        }
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemRelease(handle));
        throw;
    }
}

static int64_t import(const std::vector<int64_t>& words, const int64_t& num_bytes) {
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
    } catch (...) {
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemRelease(handle));
        throw;
    }
    track_mapping(address, Mapping(aligned, handle));
    return static_cast<int64_t>(address);
}

static void free_buffer(const int64_t& data_ptr) {
    Mapping mapping{};
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_mappings.find(static_cast<CUdeviceptr>(data_ptr));
        DG_HOST_ASSERT(it != g_mappings.end());
        mapping = it->second;
        g_mappings.erase(it);
    }
    // The caller synchronizes (SymmBuffer.destroy runs after a group barrier)
    DG_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemUnmap(data_ptr, mapping.num_bytes));
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemRelease(mapping.handle));
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemAddressFree(data_ptr, mapping.num_bytes));
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
static int64_t import(const std::vector<int64_t>&, const int64_t&) {
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
    m.def("fabric_symm_buffer_import(int[] desc, int num_bytes) -> int");
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
