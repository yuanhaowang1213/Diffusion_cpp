/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "ImgBase.h"
#include "saiga/core/util/ProgressBar.h"

ImgBase::ImgBase(std::string _scene_dir, std::shared_ptr<CombinedParams> params1)
{
    scene_path = std::filesystem::canonical(_scene_dir).string();
    scene_name = std::filesystem::path(scene_path).filename();
    params = params1;
    std::cout << ConsoleColor::BLUE ;
    log_ofs << "=============================================" << std::endl;
    log_ofs << "Scene Base " << std::endl;
    log_ofs << " Name         " << scene_name << std::endl;
    log_ofs << " Path         " << scene_path << std::endl;
    SAIGA_ASSERT(!scene_name.empty());
    CHECK(std::filesystem::exists(scene_path));
    CHECK(std::filesystem::exists(scene_path + "/datasets.ini"));

    auto file_slice_names              = scene_path + "/images.txt";
    CHECK(std::filesystem::exists(file_slice_names));
    dataset_params =  DatasetParams(scene_path + "/datasets.ini");

    image_dir = dataset_params.sequence_dir;
    std::vector<std::string> images;
    if(std::filesystem::exists(file_slice_names))
    {
        std::ifstream strm(file_slice_names);
        std::string line;
        while(std::getline(strm,line))
        {
            images.push_back(line);
        }
    }
    log_ofs << "frame size " << images.size()  << std::endl;
    int n_images = images.size();
    imgs.resize(n_images);
    for(int i = 0; i < n_images; ++i)
    {
        auto img = std::make_shared<UnifiedImg>();
        if(!images.empty()) img->img_file = images[i];
        imgs[i] = img;
    }
    log_ofs << " Input frame size " << imgs.size() << std::endl;
    log_ofs << "=============================================" << std::endl;
    std::cout << ConsoleColor::RESET ; 
}


