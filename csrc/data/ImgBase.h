/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/image/freeimage.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/torch/ImageTensor.h"
#include "saiga/core/util/ConsoleColor.h"


#include <torch/script.h>

#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include "Settings.h"
#include "utils/utils.h"

struct UnifiedImg
{
    virtual ~UnifiedImg() {}
    torch::Tensor img;
    std::vector<torch::Tensor> imgs;
    std::string img_file;
};

struct ImgBase
{
    public:
        ImgBase(std::string _scene_dir, std::shared_ptr<CombinedParams> params1);
        std::vector<std::shared_ptr<UnifiedImg>> imgs;
        std::vector<int> train_indices, test_indices, validate_indices;
        std::shared_ptr<CombinedParams> params;
        DatasetParams dataset_params;
        std::string scene_path;
        std::string scene_name;
        std::string image_dir;
        int num_channels    = 1;
        int D               = 2;
        void SaveCheckpoint(const std::string& dir) {};
        void LoadCheckpoint(const std::string& dir) {};
};

