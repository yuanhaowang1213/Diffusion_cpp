/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once

#include "Settings.h"
#include "structure/HelperStructs.h"
#include <iostream>
#include <cstdlib>
#include "globalenv.h"
#include "ImgBase.h"
using namespace Saiga;



struct ImageData
{
    int scene_id = 0;
    ImageList images;
    int num_input() {return images.image.size(0);}
};

class SequentialImageDataset : public torch::data::BatchDataset<SequentialImageDataset, ImageData>
{
    public:
        SequentialImageDataset(std::vector<int> indices, std::shared_ptr<ImgBase> scene,
                            std::shared_ptr<CombinedParams> params)
            : indices(indices), scene(scene),  params(params)
        {
            total_num_Images = indices.size();
            std::cout << "num Images " << total_num_Images << std::endl;            
            CHECK_GT(total_num_Images, 0);
            image_dir = scene->image_dir;

        }

        virtual torch::optional<size_t> size() const override {return total_num_Images;}
        virtual ImageData get_batch(torch::ArrayRef<size_t> indices);
    
    private:
        std::vector<int> indices;
        std::shared_ptr<ImgBase> scene;
        std::shared_ptr<CombinedParams> params;
        // int out_w;
        // int num_Images;
        int total_num_Images;
        std::string image_dir;
};

