/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/torch/TorchHelper.h"

#include <torch/torch.h>

#include "Settings.h"

class EMA
{
    public: EMA(std::vector<torch::Tensor>  model_parameters, float ema_rate, int ema_start = 2000)
            : ema_rate(ema_rate), ema_start(ema_start)
    {
        for(auto p : model_parameters)
        {
            {
                ema_shadow_.push_back(p.clone().to(torch::kCUDA));
            }
        }
        step = 0;
    }
    void update(std::vector<torch::Tensor>  model_parameters)
    {

        #pragma omp parallel for
        for(int count = 0; count < model_parameters.size(); ++count)
        {
            {

                if(step > ema_start)
                {
                    auto new_average = (1.0 - ema_rate )* model_parameters[count] + ema_rate * ema_shadow_[count];
                    ema_shadow_[count] = new_average;
                }
                else
                {
                    ema_shadow_[count] = model_parameters[count];

                }

            }
        }
        step++;
    }

    void apply_shadow(std::vector<torch::Tensor> model_parameters)
    {
        int count = 0;
        for(auto & p : model_parameters)
        {
            {
                ema_backup_.push_back(p.clone().to(torch::kCPU));
                p = ema_shadow_[count].to(p.device());
                count+=1;
            }
        }
    }

    void restore(std::vector<torch::Tensor> model_parameters)
    {
        int count = 0;
        for(auto & p : model_parameters)
        {
            {
                p = ema_backup_[count].to(p.device());
                count+=1;
            }
        }
    }
    void SaveCkpt(std::string savedir)
    {
        torch::save(ema_shadow_, savedir + "/ema_shadow.pt");

    }
    void LoadCkpt(std::string savedir)
    {
        if(std::filesystem::exists(savedir+"/ema_shadow.pt"))
        {
            log_ofs << "========Preloading ema ========" << std::endl;
            torch::load(ema_shadow_, savedir+"/ema_shadow.pt");
        }
    }

    private:
        int step;
        std::vector<torch::Tensor> ema_shadow_;
        std::vector<torch::Tensor> ema_backup_;
        int ema_start;
        float ema_rate;
};
