/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "geometry.h"
#include "saiga/vision/torch/ImageTensor.h"

#include "saiga/vision/torch/ColorizeTensor.h"

MriNeuralGeometry::MriNeuralGeometry(int num_channels, int D, std::shared_ptr<CombinedParams> params)
    : NeuralGeometry(num_channels, D, params)
{
    std::cout << ConsoleColor::BLUE ;
    log_ofs  << "=== Neural Geometry is " << std::endl;
    
    log_ofs << "Last Activation " << params->net_params.last_activation_function << std::endl;

    // TO DO
    // Add decoder

}

void MriNeuralGeometry::AddParametersToOptimizer() {}

void MriNeuralGeometry::SaveCkpt(const std::string& dir)
{
    log_ofs << "========Saving Optimizer  ========" << std::endl;

    if(params->train_params.mri_op == "adam")
    {
        torch::save(*optimizer_adam, dir + "/adam_optimizer.pt");
    }
    else if (params->train_params.mri_op == "rms")
    {
        torch::save(*optimizer_rms, dir+"/rms_optimizer.pt");
    }
    else
    {
        CHECK(false) << "Not supporting optimization method saving" << std::endl;
    }
}

void MriNeuralGeometry::LoadCkpt(const std::string& dir)
{
    log_ofs << "========Preloading Optimizer ========" << std::endl;
    if(!std::filesystem::is_empty(dir))
    {
        if(params->train_params.mri_op == "adam")
        {
            CHECK(std::filesystem::exists(dir + "/adam_optimizer.pt"));
            torch::load(*optimizer_adam, dir + "/adam_optimizer.pt");
        }
        else if (params->train_params.mri_op == "rms")
        {
            CHECK(std::filesystem::exists(dir+"/rms_optimizer.pt"));
            torch::load(*optimizer_rms, dir+"/rms_optimizer.pt");
        }
        else
        {
            CHECK(false) <<"Not support optimization method loading " << std::endl;
        }
    }
}


