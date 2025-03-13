/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/assert.h"
#include "saiga/cuda/imgui_cuda.h"

#include "Settings.h"

#include "saiga/core/image/freeimage.h"
#include "saiga/vision/torch/ColorizeTensor.h"
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/ImageTensor.h"
#include <torch/nn/parallel/data_parallel.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>
#include <torch/script.h>

using namespace Saiga;

class NeuralGeometry : public torch::nn::Module
{
    public:
    NeuralGeometry(int num_channels, int D, std::shared_ptr<CombinedParams> params)
        : num_channels(num_channels), D (D), params(params)
    {
        float base_lr = 0.0;
        if(params->train_params.mri_op == "adam")
        {
            base_lr = params->train_params.lr_mri_adam;
        }
        else if(params->train_params.mri_op == "rms")
        {
            base_lr = params->train_params.lr_mri_rms;
        }
        else
        {
            CHECK(false) << "Not supporting optimization method " << std::endl;
        }
    }

    virtual void train(int epoch_id, bool on)
    {
        torch::nn::Module::train(on);
        c10::cuda::CUDACachingAllocator::emptyCache();
        if(on)
        {
            if(!optimizer_adam && !optimizer_sgd)
            {
                CreateGeometryOptimizer();
            }
            if(optimizer_adam) optimizer_adam->zero_grad();
            if(optimizer_sgd)  optimizer_sgd->zero_grad();
            if(optimizer_rms)  optimizer_rms->zero_grad();
            if(optimizer_decoder) optimizer_decoder->zero_grad();
        }
    }

    void ResetGeometryOptimizer() {CreateGeometryOptimizer();}

    void CreateGeometryOptimizer()
    {
        optimizer_adam = 
            std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>(), torch::optim::AdamOptions().lr(params->train_params.lr_mri_adam).weight_decay(params->train_params.weight_decay));
        optimizer_rms = 
            std::make_shared<torch::optim::RMSprop>(std::vector<torch::Tensor>(), torch::optim::RMSpropOptions().lr(params->train_params.lr_mri_rms).weight_decay(params->train_params.weight_decay));
        optimizer_sgd =
            std::make_shared<torch::optim::SGD>(std::vector<torch::Tensor>(), torch::optim::SGDOptions(10).weight_decay(params->train_params.weight_decay));

        AddParametersToOptimizer();
    }

    virtual void PrintGradInfo(int epoch_id, TensorBoardLogger* logger) {}

    void OptimizerStep(int epoch_id)
    {

        if(optimizer_adam)
        {
            optimizer_adam->step();
            optimizer_adam->zero_grad();
        }
        if(optimizer_rms)
        {
            optimizer_rms->step();
            optimizer_rms->zero_grad();
        }
        if(optimizer_sgd)
        {
            optimizer_sgd->step();
            optimizer_sgd->zero_grad();
        }
        if(optimizer_decoder)
        {
            optimizer_decoder->step();
            optimizer_decoder->zero_grad();
        }
    }

    void UpdateLearingRate(double factor)
    {
        if(optimizer_adam) UpdateLR(optimizer_adam.get(), factor);
        if(optimizer_rms)  UpdateLR(optimizer_rms.get(), factor);
        if(optimizer_sgd)  UpdateLR(optimizer_sgd.get(), factor);

    }

    void UpdateDecoderLearingRate(double factor)
    {
        if(optimizer_decoder) UpdateLR(optimizer_decoder.get(), factor);
    }

    std::vector<double> getDecoderLearningRate()
    {
        // auto optimizer_lr = optimizer_decoder.get();
        std::vector<double> learning_rate;
        for (auto& pg : optimizer_decoder.get()->param_groups())
        {
            auto opt_rms = dynamic_cast<torch::optim::RMSpropOptions*>(&pg.options());
            if (opt_rms)
            {
                // opt_rms->lr() = opt_rms->lr() * factor;
                learning_rate.push_back(opt_rms->lr());
            }
        }

        return learning_rate;
    }

    std::vector<double> GetNetworkLearningRate()
    {
        std::vector<double> learning_rate;

        if(optimizer_adam)
        {
            for(auto& pg: optimizer_adam.get()->param_groups())
            {
                // if(!pg.grad().defined())
                // continue;
                auto options = static_cast<torch::optim::AdamOptions &>(pg.options());
                learning_rate.push_back(options.get_lr());
                // update learing rate 
                // options.lr(options.lr()*(1.0-rate));
            }
        }
        if(optimizer_rms)
        {
            for(auto& pg: optimizer_rms.get()->param_groups())
            {
                // if(!pg.grad().defined())
                // continue;
                auto options = dynamic_cast<torch::optim::RMSpropOptions *>(&pg.options());
                learning_rate.push_back(options->get_lr());
            }
        }
        return learning_rate;
    }
    // float GetUpdateLr(int step) {return lrschedule->update_lr(step);}    
    public:
    std::shared_ptr<torch::optim::Adam> optimizer_decoder;

    std::shared_ptr<torch::optim::Adam> optimizer_adam;
    std::shared_ptr<torch::optim::SGD> optimizer_sgd;
    std::shared_ptr<torch::optim::RMSprop> optimizer_rms;

    protected:

    int num_channels;
    int D;
    std::shared_ptr<CombinedParams> params;
    virtual void AddParametersToOptimizer() = 0;
};

class MriNeuralGeometry : public NeuralGeometry
{
    public:
        MriNeuralGeometry(int num_channels, int D, std::shared_ptr<CombinedParams> params);

    virtual void to(torch::Device device, bool non_blocking = false)
    {
        NeuralGeometry::to(device, non_blocking);
    }


    void SavePython(std::string load_file, std::string save_file)
    {
        torch::jit::script::Module modulepython = torch::jit::load(load_file);
        modulepython.save(save_file);
    }
    virtual torch::Tensor sampling(const torch::Tensor & inputs, std::function<torch::Tensor(const torch::Tensor &)> A, std::function<torch::Tensor(const torch::Tensor &)> Ap) {return {};}
    virtual void Broadcast_back() {};
    virtual std::vector<double> Compute_Loss( torch::Tensor& sequence, torch::Tensor& slice, torch::Tensor  weight = torch::Tensor() , torch::Tensor noise = torch::Tensor()) {return {};}

    virtual torch::Tensor Eval_Image(torch::Tensor& sequence, torch::Tensor& slice, torch::Tensor weight = torch::Tensor(), torch::Tensor noise = torch::Tensor() ) {return {};}
    // save checkpoint and network
    // load checkpoint and network
    virtual void SaveCkpt(const std::string& dir);
    virtual void LoadCkpt(const std::string& dir);

    virtual void ema_apply_shadow() {};
    virtual void ema_restore() {};
    protected:
    // The per-sample weight is multiplied to the raw sample output.

    virtual void AddParametersToOptimizer();

    torch::Device device = devices.front();
    torch::autograd::Scatter scatter = torch::autograd::Scatter(devices, torch::nullopt, 0);
    int device_count = devices.size();
    public:

    Saiga::CUDA::CudaTimerSystem* timer = nullptr;
};