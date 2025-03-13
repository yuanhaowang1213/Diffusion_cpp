/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "geometry_diffun.h"

Geometry_DiffUn::Geometry_DiffUn(int num_channels, int D, std::shared_ptr<CombinedParams> params)
            : MriNeuralGeometry(num_channels, D, params)
{
    log_ofs << "Type: Geometry DiffUnfolding with ResUnet " << std::endl;


    log_ofs << "input channel " << params->net_params.unet_input_ch << " output channel "<< params->net_params.unet_output_ch <<std::endl;

    in_chans_ = params->net_params.unet_input_ch + params->net_params.unet_output_ch;
    cascades_ = torch::nn::ModuleList();
    for(int i =0; i< params->net_params.deep_unfolding_iter; ++i)
    {
        cascades_->push_back(EDM_UNETModel(in_chans_,params->net_params.unet_output_ch, params ));
    }
    register_module("cascades_", cascades_);
    for (auto &module : *cascades_)
    {
        module->to(device);  // Move each submodule to the device
    }

    ema_ = std::make_shared<EMA>(cascades_->parameters(), params->train_params.ema_rate); 

    num_steps_ =   params->train_params.diffusion_steps;
    sigma_min_log_ = std::log(sigma_min_ + epsilon_t);
    sigma_max_log_ = std::log(sigma_max_ + epsilon_t);
    std::vector<int64_t> empty_kernel;

    // Noise parameter model
    noise_param = torch::nn::Sequential(
        MPFourier(logvar_channels_),
        MPConv2(logvar_channels_, 2, empty_kernel)
    );
    register_module("noise_param", noise_param);

    // Load pre-trained weights for noise parameter
    std::string weight_path = "";
    if(std::filesystem::exists(params->net_params.weight_net_path))
    {
        log_ofs<<"loading weight path " << params->net_params.weight_net_path << std::endl;
        torch::load(noise_param, params->net_params.weight_net_path);
    }
    // torch::load(noise_param, weight_path);
    std::cout << "To do need to load the noise params \n" << std::endl;

    // Register parameters
    dc_weight = register_parameter("dc_weight", torch::ones({params->net_params.deep_unfolding_iter}));
    dn_weight = register_parameter("dn_weight", torch::ones({params->net_params.deep_unfolding_iter}));

    sigma_min_log_ = std::log(sigma_min_ + epsilon_t);
    sigma_max_log_ = std::log(sigma_max_ + epsilon_t);



    logvar_fourier->push_back(MPFourier(logvar_channels_));
    logvar_linear->push_back(MPConv(logvar_channels_, 1, empty_kernel));
    register_module("logvar_fourier",logvar_fourier);
    register_module("logvar_linear",logvar_linear);

}

void Geometry_DiffUn:: AddParametersToOptimizer()
{
    MriNeuralGeometry::AddParametersToOptimizer();
    if(params->train_params.mri_op == "adam")
    {
        log_ofs << "Optimizing Using Adam optimizer " << std::endl;
        optimizer_adam->add_param_group({cascades_->parameters(),
            std::make_unique<torch::optim::AdamOptions>(params->train_params.lr_mri_adam)});
    }

    else if(params->train_params.mri_op == "rms")
    {
        log_ofs << "Optimizing Using RMS " << std::endl;
        optimizer_rms->add_param_group({cascades_->parameters(),
            std::make_unique<torch::optim::RMSpropOptions>(params->train_params.lr_mri_rms)});
    }
    else
    {
        CHECK(false) << "Not supporting optimization method " << std::endl;
    }
}



void Geometry_DiffUn::SaveCkpt(const std::string& dir)
{
    MriNeuralGeometry::SaveCkpt(dir);

    log_ofs << "========Saving network ========" << std::endl;
    torch::save(cascades_, dir+"/attention-unet.pt" );

}
void Geometry_DiffUn::LoadCkpt(const std::string& dir)
{
    if(!std::filesystem::is_empty(dir))
    {
        MriNeuralGeometry::LoadCkpt(dir);
        
        if(std::filesystem::exists(dir + "/attention-unet.pt"))
        {
            log_ofs << "========Preloading network ========" << std::endl;
            // torch::load(holder, dir+"/attention-unet.pt");
            torch::load(cascades_, dir+"/attention-unet.pt");


        }

    }


}


std::vector<double> Geometry_DiffUn::Compute_Loss(torch::Tensor& sequence, torch::Tensor& slice, torch::Tensor weight= torch::Tensor(), torch::Tensor noise = torch::Tensor())
{
 
    std::vector<double> loss_value;
 

    return loss_value;

}

torch::Tensor Geometry_DiffUn::Eval_Image(torch::Tensor& sequence, torch::Tensor& slice, torch::Tensor weight = torch::Tensor(), torch::Tensor noise = torch::Tensor())
{


    torch::Tensor result;

    slice = slice.to(devices.front());

    CHECK_EQ(result.sizes(), slice.sizes());
    return result;
}



torch::Tensor Geometry_DiffUn::model_forward(EDM_UNETModelImpl & model, torch::Tensor sigma, torch::Tensor x, 
                            torch::Tensor y= torch::Tensor()) 
                            
{
    bool complex_flag = x.is_complex();
    if (complex_flag) {
        x = torch::view_as_real(x);
        x = x.permute({0, 4, 2, 3, 1});
        x = x.slice(-1,0,1);
    }
    x = x.to(torch::kFloat32);
    sigma = sigma.to(torch::kFloat32).reshape({-1, 1, 1, 1});
    // torch::Tensor class_labels = label_dim == 0 ? torch::Tensor() : torch::zeros({1, label_dim}, x.device());

    torch::Dtype dtype =  torch::kFloat32;
    torch::Tensor c_skip = sigma_data_ * sigma_data_ / (sigma * sigma + sigma_data_ * sigma_data_);

    torch::Tensor c_out = sigma * sigma_data_ / torch::sqrt(sigma * sigma + sigma_data_ * sigma_data_) ;

    torch::Tensor c_in = 1 / torch::sqrt(sigma * sigma + sigma_data_ * sigma_data_);


    torch::Tensor c_noise = sigma.log() / 4;

    torch::Tensor model_input = y.defined() ? torch::cat({c_in * x, y}, 1) : c_in * x;

    torch::Tensor F_x = model.forward(model_input.to(dtype), c_noise.flatten().to(dtype));

    torch::Tensor D_x = c_skip * x + c_out * F_x.to(torch::kFloat32);

    return D_x;
}

torch::Tensor Geometry_DiffUn::sampling(const torch::Tensor & inputs, std::function<torch::Tensor(const torch::Tensor &)> A,
                            std::function<torch::Tensor(const torch::Tensor &)> Ap)
{
    // Time step discretization
    torch::Tensor step_indices = torch::arange(num_steps_, torch::TensorOptions().dtype(torch::kFloat64));
    torch::Tensor t_steps = torch::pow(
        std::pow(sigma_max_, 1 / rho_) + step_indices / (num_steps_ - 1) * 
        (std::pow(sigma_min_, 1 / rho_) - std::pow(sigma_max_, 1 / rho_)), rho_
    );
    // Append t_N = 0 to t_steps
    t_steps = torch::cat({t_steps, torch::zeros({1}, t_steps.options())}, 0);
    // Initialize `x_next`
    torch::Tensor x_next = torch::randn_like(Ap(inputs)) * t_steps.data_ptr<double>()[0];
    std::vector<torch::Tensor> x_next_stack = {x_next.unsqueeze(1)};
    // Determine model input based on degradation type
    torch::Tensor model_input;
    // if (params->net_params.deg.find("inpainting") != std::string::npos || params->net_params.deg.find("sr") != std::string::npos || params->dataset_name.find("fastmri") != std::string::npos) 
    if (params->net_params.deg.find("inpainting") != std::string::npos || params->net_params.deg.find("sr") != std::string::npos ) 
    {
        // std::cout <<"test sr " << TensorInfo(inputs) << std::endl;
        model_input = Ap(inputs);

    } else {
        model_input = inputs;
    }
    // Iterate through timesteps
    for (int i = 0; i < num_steps_ - 1; ++i) {


        double t_cur = t_steps.data_ptr<double>()[i];
        double t_next = t_steps.data_ptr<double>()[i+1];

        // Thresholding to zero out model input for small `t_cur`
        if (t_cur < thres_) {
            model_input = torch::zeros_like(model_input);
        }
        torch::Tensor x_cur = x_next;
        float gamma = (S_min_ <= t_cur && t_cur<= S_max_)
                      ? std::min(S_churn_ / num_steps_,float( std::sqrt(2) - 1))
                      : 0;

        double t_hat = (t_cur + gamma * t_cur);



        torch::Tensor x_hat = x_cur + (std::sqrt((t_hat * t_hat - t_cur * t_cur)) * S_noise_ * torch::randn_like(x_cur));

        // Compute noise level and weight_sigma
        torch::Tensor sigma_to_est = t_hat*torch::ones({1},model_input.options()).reshape({-1,1,1,1}).flatten() / sigma_max_;

        torch::Tensor weight_sigma = noise_param->forward(sigma_to_est.to(torch::kFloat32));

        // Iteratively refine results using the cascades
        torch::Tensor results = x_hat.clone().detach();
        int indx = 0;
        for (auto &module : *cascades_)
        {
            EDM_UNETModelImpl& cascade = *module->as<EDM_UNETModelImpl>();  // Correctly extract the EDM_UNETModelImpl

            // Compute data consistency term
            torch::Tensor dc = Ap(A(results) - inputs);

            torch::Tensor dc_term = dc / dc.square().sum().sqrt();
            // .square().sum().sqrt();

            // Model inference
            torch::Tensor output = model_forward(cascade, t_hat*torch::ones({1},model_input.options()), results, model_input);

            // Weighted update step

            results = results - dc_weight.slice(0,indx,indx+1) * (weight_sigma.slice(1,0,1).reshape({-1,1,1,1}) * dc_term +
                                                    dn_weight.slice(0,indx,indx+1) * weight_sigma.slice(1,1,2).reshape({-1,1,1,1}) * (results - output));

            indx+=1;
        }

        // Compute denoised estimate
        torch::Tensor denoised = results.clone().detach();
        torch::Tensor d_cur = (x_hat - denoised) / t_hat;
        x_next = x_hat + (t_next - t_hat) * d_cur;

    }

    // Concatenate and return results
    return torch::cat(x_next_stack, 1);
}