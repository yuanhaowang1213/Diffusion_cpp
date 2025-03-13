/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include <torch/torch.h>
#include "log_file.h"

inline std::vector<torch::Device> devices = std::vector<torch::Device>();

inline output_stream log_ofs;
