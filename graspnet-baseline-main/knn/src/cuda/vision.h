#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cstdint>

void knn_device(
    float* ref_dev,
    int64_t ref_width,
    float* query_dev,
    int64_t query_width,
    int64_t height,
    int64_t k,
    float* dist_dev,
    int64_t* ind_dev,
    cudaStream_t stream
);