/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "data/Dataloader.h"

ImageData SequentialImageDataset::get_batch(torch::ArrayRef<size_t> sample_indices)
{
    std::vector<ImageList> image_list;
    torch::set_num_threads(1);
    // for(auto index : sample_indices)
    #pragma omp parallel for
    for(int i = 0; i < sample_indices.size();++i)
    {
        auto index = sample_indices[i];
        ImageList result;
        std::shared_ptr<UnifiedImg> img = scene->imgs[index];
        TemplatedImage<ucvec3> image_3;
        image_3.load(image_dir + img->img_file);

        auto view = image_3.getImageView();
        if (!view.valid()) {  // Check if the image view is valid (pseudo-code)
            std::cerr << "Invalid ImageView!" << std::endl;
            continue;
        }
        auto image = ImageViewToTensor(view);
        result.image = image.unsqueeze(0); 

   
        image_list.push_back(result);
    }
    auto device1 = torch::kCUDA;
    ImageData sample_data;
    sample_data.images = ImageList(image_list);
    sample_data.images.to(device1);
    return sample_data;
}
