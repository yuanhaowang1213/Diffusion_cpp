/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#define CUDA_NDEBUG

#include "saiga/cuda/cuda.h"
#include "saiga/vision/torch/CudaHelper.h"
#include "saiga/vision/torch/EigenTensor.h"
#include "utils/svd_operators.h"
static __global__ void MaskFill(StaticDeviceTensor<float,1> singulars, StaticDeviceTensor<float,1> factors)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id > singulars.sizes[0]) return;
    if(singulars(id) == 0)
    factors(id) = 0;
}

void A_function::MaskZeroFill(torch::Tensor singulars, torch::Tensor factors)
{    
    TORCH_CHECK(singulars.device().is_cuda(), "MaskZero: input tensor must be on CUDA!");
    int size = singulars.numel();
    int threads = 128;
    int blocks = (size + threads - 1)/threads;
    ::MaskFill<<<blocks, threads>>>(singulars, factors);
    CUDA_SYNC_CHECK_ERROR();
}