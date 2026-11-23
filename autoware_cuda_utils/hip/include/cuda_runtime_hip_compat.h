// Copyright 2026 The Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The CUDA runtime names used by autoware_cuda_utils, mapped to their HIP equivalents.

#ifndef CUDA_RUNTIME_HIP_COMPAT_H_
#define CUDA_RUNTIME_HIP_COMPAT_H_

// Types
#ifndef cudaError_t
#define cudaError_t hipError_t
#endif
#ifndef cudaStream_t
#define cudaStream_t hipStream_t
#endif
#ifndef cudaMemPool_t
#define cudaMemPool_t hipMemPool_t
#endif
#ifndef cudaMemPoolProps
#define cudaMemPoolProps hipMemPoolProps
#endif
#ifndef cudaPointerAttributes
#define cudaPointerAttributes hipPointerAttribute_t
#endif

// Constants
#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#endif
#ifndef cudaErrorInvalidValue
#define cudaErrorInvalidValue hipErrorInvalidValue
#endif
#ifndef cudaHostAllocDefault
#define cudaHostAllocDefault hipHostAllocDefault
#endif
#ifndef cudaMemAllocationTypePinned
#define cudaMemAllocationTypePinned hipMemAllocationTypePinned
#endif
#ifndef cudaMemLocationTypeDevice
#define cudaMemLocationTypeDevice hipMemLocationTypeDevice
#endif
#ifndef cudaMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#endif
#ifndef cudaMemoryTypeHost
#define cudaMemoryTypeHost hipMemoryTypeHost
#endif
#ifndef cudaMemoryTypeDevice
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#endif
#ifndef cudaStreamDefault
#define cudaStreamDefault hipStreamDefault
#endif
#ifndef cudaStreamNonBlocking
#define cudaStreamNonBlocking hipStreamNonBlocking
#endif

// Errors and devices
#ifndef cudaGetErrorName
#define cudaGetErrorName hipGetErrorName
#endif
#ifndef cudaGetErrorString
#define cudaGetErrorString hipGetErrorString
#endif
#ifndef cudaGetDeviceCount
#define cudaGetDeviceCount hipGetDeviceCount
#endif
#ifndef cudaPointerGetAttributes
#define cudaPointerGetAttributes hipPointerGetAttributes
#endif

// Memory
#ifndef cudaMalloc
#define cudaMalloc hipMalloc
#endif
#ifndef cudaFree
#define cudaFree hipFree
#endif
#ifndef cudaMemsetAsync
#define cudaMemsetAsync hipMemsetAsync
#endif
#ifndef cudaHostAlloc
#define cudaHostAlloc hipHostAlloc
#endif
#ifndef cudaFreeHost
#define cudaFreeHost hipHostFree
#endif
#ifndef cudaMemPoolCreate
#define cudaMemPoolCreate hipMemPoolCreate
#endif
#ifndef cudaMemPoolSetAttribute
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#endif
#ifndef cudaMallocFromPoolAsync
#define cudaMallocFromPoolAsync hipMallocFromPoolAsync
#endif
#ifndef cudaFreeAsync
#define cudaFreeAsync hipFreeAsync
#endif

// Streams
#ifndef cudaStreamCreate
#define cudaStreamCreate hipStreamCreate
#endif
#ifndef cudaStreamCreateWithFlags
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#endif
#ifndef cudaStreamDestroy
#define cudaStreamDestroy hipStreamDestroy
#endif
#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#endif
#ifndef cudaStreamQuery
#define cudaStreamQuery hipStreamQuery
#endif
#ifndef cudaStreamGetFlags
#define cudaStreamGetFlags hipStreamGetFlags
#endif

#endif  // CUDA_RUNTIME_HIP_COMPAT_H_
