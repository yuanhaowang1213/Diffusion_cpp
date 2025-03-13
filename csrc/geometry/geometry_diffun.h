/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include <torch/script.h>

#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include "Settings.h"
#include "data/ImgBase.h"
#include "geometry.h"
// #include "unet.h"
#include "nnbasic.h"

#include "ema.h"

class Geometry_DiffUn: public MriNeuralGeometry
{
    public:
        Geometry_DiffUn(int num_channels, int D, std::shared_ptr<CombinedParams>params);
        void SaveCkpt(const std::string& dir) override;
        void LoadCkpt(const std::string& dir) override;



    protected:
        void AddParametersToOptimizer() override;

        virtual std::vector<double> Compute_Loss(torch::Tensor& sequence, torch::Tensor& slice , torch::Tensor  weight, torch::Tensor noise ) override;

        virtual torch::Tensor Eval_Image(torch::Tensor & sequence, torch::Tensor& slice, torch::Tensor weight, torch::Tensor noise ) override;


        torch::Tensor model_forward(EDM_UNETModelImpl & model, torch::Tensor sigma, torch::Tensor x, torch::Tensor y );

        torch::nn::ModuleList cascades_;
        std::shared_ptr<EMA> ema_;

        virtual torch::Tensor sampling(const torch::Tensor & inputs, std::function<torch::Tensor(const torch::Tensor &)> A,
                            std::function<torch::Tensor(const torch::Tensor &)> Ap) override;


        virtual void ema_apply_shadow () override{ema_->apply_shadow(cascades_->parameters());} 
        virtual void ema_restore () override{ema_->restore(cascades_->parameters());} 

    public:
        int in_chans_ ;
        float S_churn_              = 0;
        float S_min_                = 0.0;
        float S_max_ = std::numeric_limits<float>::infinity();
        float S_noise_              = 1.0;
        int num_steps_;
        float sigma_min_            = 0.002;
        float sigma_max_            = 80.0f;
        float rho_                  = 7.0f;
        float sigma_data_           = 0.5;
        float p_mean_               = -1.2;
        float p_std_                = 1.2;
        int   label_dim_            = 0;
        float epsilon_t             = 1e-6;
        float sigma_min_log_        ;
        float sigma_max_log_        ;
        float thres_                = 0.05;
        int logvar_channels_         = 128;
        torch::nn::Sequential noise_param, logvar_fourier, logvar_linear;
        torch::Tensor dc_weight, dn_weight;

};
