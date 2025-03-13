/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/image/freeimage.h"
#include "saiga/core/time/time.h"


#include "saiga/vision/torch/TrainParameters.h"
#include "tensorboard_logger.h"

#include "build_config.h"
#include "globalenv.h"
using namespace Saiga;

struct NetParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT(NetParams);

    SAIGA_PARAM_STRUCT_FUNCTIONS;

    int hidden_features             = 512;
    int hidden_layers               = 4;
    std::string geometry_type       = "exex";
    std::string last_activation_function = "softplus";

    std::string beta_schedule_type  = "linear";
    int decoder_hidden_layers       = 2;
    int decoder_hidden_features     = 64;

    float decoder_lr                = 1e-4;
    std::string decoder_activation  = "silu";
    std::string indi_time_src       = "betas";
    float indi_noise_factor         = 0.01;
    float gaussian_std              = 4;

    int deep_unfolding_iter         = 6;
    int sens_chans                  = 8;
    int sens_pools                  = 4;
    bool test_network               = false;
    float du_tau                    = 0.1;
    std::string unet_type           = "unet";

    int edm_unet_num_blocks         = 3;
    int unet_input_ch               = 10;
    int unet_output_ch              = 2;

    std::string jit_module_path     = "";

    int img_size                    = 256;
    std::string deg                 = "sr_bicubic";
    std::string weight_net_path     = "";
    std::string mask_type           = "";
    int scale                       = 4;

    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        SAIGA_PARAM(hidden_features);
        SAIGA_PARAM(hidden_layers);
        SAIGA_PARAM(geometry_type);
        SAIGA_PARAM(last_activation_function);
        SAIGA_PARAM(beta_schedule_type);
        SAIGA_PARAM(indi_time_src);
        SAIGA_PARAM(indi_noise_factor);
        SAIGA_PARAM(gaussian_std);
        SAIGA_PARAM(decoder_lr);
        SAIGA_PARAM(decoder_activation);
        SAIGA_PARAM(decoder_hidden_layers);
        SAIGA_PARAM(decoder_hidden_features);
        SAIGA_PARAM(deep_unfolding_iter);
        SAIGA_PARAM(sens_chans);
        SAIGA_PARAM(sens_pools);
        SAIGA_PARAM(test_network);
        SAIGA_PARAM(du_tau);
        SAIGA_PARAM(unet_type);
        SAIGA_PARAM(unet_input_ch);
        SAIGA_PARAM(unet_output_ch);
        SAIGA_PARAM(edm_unet_num_blocks);
        SAIGA_PARAM(jit_module_path);
        SAIGA_PARAM(img_size);
        SAIGA_PARAM(deg);
        SAIGA_PARAM(weight_net_path);
        SAIGA_PARAM(mask_type);
    }
};

struct DatasetParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT(DatasetParams);

    SAIGA_PARAM_STRUCT_FUNCTIONS;

    std::string sequence_dir = "";
    std::string slice_dir    = "";
    std::string mask_dir     = "";
    std::string data_type    = "";

    float mri_max            = 1;
    float mri_min            = -1;
    int train_slice          = -1;

    std::string coil_weight_file = "";

    vec3 dimension_min       = vec3(32, 0, 0);
    vec3 dimension_max       = vec3(223, 0, 0);

    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        SAIGA_PARAM(sequence_dir);
        SAIGA_PARAM(slice_dir);
        SAIGA_PARAM(mask_dir);
        SAIGA_PARAM(data_type);
        SAIGA_PARAM(mri_max);
        SAIGA_PARAM(mri_min);
        SAIGA_PARAM(train_slice);
        SAIGA_PARAM(coil_weight_file);
        SAIGA_PARAM_LIST(dimension_min, ' ');
        SAIGA_PARAM_LIST(dimension_max, ' ');
    }
};

struct MyTrainParams : public TrainParams
{


    MyTrainParams() {}
    MyTrainParams(const std::string& file) { Load(file); }

    using ParamStructType = MyTrainParams;

    SAIGA_PARAM_STRUCT_FUNCTIONS;

    int train_num_threads               = 16;
    std::string scene_dir               = " ";
    std::vector<std::string> scene_name = {"pepper"}; // Experiment dir
    std::vector<std::string> validate_echo      = {"0"};
    std::string split_name              = "exp_uniform_50"; //experiment list dir

    std::string coil_weight_file        = ""; 

    std::string test_complex_space      = "real_imag"; // real_imag or mag_phase
    int test_direction                  = 0; 
    // 0 choose yz direction 
    // 1 choose xz direction
    // 2 choose xy direction
    std::string coil_sampling_mask_flile = "";

    int validate_direction              = 0;
    bool validate_on                    = false; // not for validate, on is for validate

    std::string mri_op                  = "adam"; // optimizer
    std::string data_type               = ""; // qmri raw data or processed data
    std::string exp_dir_name            = "Experiments";
    bool use_abs_expdir                 = false; 
    std::string experiment_dir          = "";
    int per_node_batch_size             = 64;

    int output_volume_size              = 64;
    bool optimize_volume                = true;
    float lr_mri_rms              = 0.01;
    float lr_mri_adam             = 0.01;
    float lr_mri_sensemap                 = 0.1f;
    float lr_ssim_scale             = 0.1f;
    // learning schedule 
    std::string lr_schedule_name    = "linear"; // linear, cosine, const
    bool use_lrwarmup               = true;


    float loss_mse_scale                = 1.0f;
    float loss_l1_scale                 = 0.0f;
    float loss_tv                       = 0.001;
    float test_tv                       = -1;
    std::string loss_type           = "mse"; // mse, l1, ssim


    float weight_decay                  = 0.0;
    float ema_rate                      = 0.9999;

    // mask settings 
    int center_lines                    = 40;
    float acceleration_rate             = 4;
    vec2 mask_sizes         = {256,192};



    // UNET parameters 
    int model_channels                  = 128;
    int num_res_blocks                  = 2;
    std::vector<std::string> attention_resolution   = {"20"};
    float dropout                       = 0;
    std::vector<std::string> channel_mult       = {};
    int num_heads                       = 4;
    int num_heads_channels              = -1;
    int num_heads_upsample              = -1;
    bool use_scale_shift_norm           = true;
    bool resblock_updown                = true; 
    bool use_new_attention_order        = false;
    int num_classes                     = -1;

    // image max size on all dim
    int max_dim                         = 256;
    int qmri_time_step                  = 10;
    // Diffusion models 
    std::string diffusion_type          = "ddpm";
    int diffusion_steps                 = 1000;
    std::string schedule_sampler        = "uniform";
    std::string noise_schedule          = "cosine";
    float beta_scale                    = 0.5;
    std::string timestep_resapcing      = "";
    bool predict_epsilon                = true;
    // bool learn_sigma                    = false;
    bool use_tilde_beta                 = false;
    bool use_corrector                  = false;
    bool rescale_timesteps              = false;
    bool model_use_ckpt                 = false;


    int input_normalize                = 1;
    bool test_input_ifft                = false;
    bool test                           = true;
    bool test_fft_before                = true;
    bool test_normalize                 = true; 
    std::string test_inputfile          = "combine_032";


    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        TrainParams::Params(it);
        SAIGA_PARAM(train_num_threads);
        SAIGA_PARAM(exp_dir_name);
        SAIGA_PARAM(use_abs_expdir);
        SAIGA_PARAM(experiment_dir);
        SAIGA_PARAM(scene_dir);
        SAIGA_PARAM_LIST(scene_name, ' ');
        SAIGA_PARAM(split_name);
        SAIGA_PARAM(coil_weight_file);
        SAIGA_PARAM_LIST(validate_echo, ' ');
        
        SAIGA_PARAM(test_direction);
        SAIGA_PARAM(test_complex_space);

        SAIGA_PARAM(coil_sampling_mask_flile);

        SAIGA_PARAM(validate_direction);
        SAIGA_PARAM(validate_on);

        SAIGA_PARAM(mri_op);
        SAIGA_PARAM(data_type);

        SAIGA_PARAM(per_node_batch_size);
        
        SAIGA_PARAM(output_volume_size);

        SAIGA_PARAM(optimize_volume);
        SAIGA_PARAM(lr_mri_rms);
        SAIGA_PARAM(lr_mri_adam);
        SAIGA_PARAM(lr_ssim_scale);

        SAIGA_PARAM(lr_mri_sensemap);

        // lr schedule 
        SAIGA_PARAM(lr_schedule_name);
        SAIGA_PARAM(use_lrwarmup);

        SAIGA_PARAM(loss_mse_scale);
        SAIGA_PARAM(loss_l1_scale);
        SAIGA_PARAM(loss_tv);
        SAIGA_PARAM(test_tv);

        SAIGA_PARAM(loss_type);

        SAIGA_PARAM(max_dim);

        SAIGA_PARAM(qmri_time_step);

        SAIGA_PARAM(weight_decay);
        SAIGA_PARAM(ema_rate);


        // mask settings 
        SAIGA_PARAM(center_lines);
        SAIGA_PARAM(acceleration_rate);
        SAIGA_PARAM_LIST(mask_sizes,' ');
        // unet default settings 
        SAIGA_PARAM(model_channels);
        SAIGA_PARAM(num_res_blocks);
        SAIGA_PARAM_LIST(attention_resolution, ' ');
        SAIGA_PARAM(dropout);
        SAIGA_PARAM_LIST(channel_mult, ' ');
        SAIGA_PARAM(num_heads);
        SAIGA_PARAM(num_heads_channels);
        SAIGA_PARAM(num_heads_upsample);
        SAIGA_PARAM(use_scale_shift_norm);
        SAIGA_PARAM(resblock_updown);
        SAIGA_PARAM(use_new_attention_order);
        SAIGA_PARAM(num_classes);

        // Diffusion model
        SAIGA_PARAM(diffusion_type);
        SAIGA_PARAM(diffusion_steps);
        SAIGA_PARAM(schedule_sampler);
        SAIGA_PARAM(noise_schedule);
        SAIGA_PARAM(beta_scale);
        SAIGA_PARAM(timestep_resapcing);
        SAIGA_PARAM(predict_epsilon);
        // SAIGA_PARAM(learn_sigma);
        SAIGA_PARAM(use_tilde_beta);
        SAIGA_PARAM(use_corrector);
        SAIGA_PARAM(rescale_timesteps);
        SAIGA_PARAM(model_use_ckpt);

        SAIGA_PARAM(input_normalize);
        SAIGA_PARAM(test_input_ifft);
        SAIGA_PARAM(test);
        SAIGA_PARAM(test_fft_before);
        SAIGA_PARAM(test_inputfile);
        SAIGA_PARAM(test_normalize);
    }
};


struct CombinedParams
{
    MyTrainParams train_params;
    NetParams net_params;

    CombinedParams() {}

    CombinedParams(const std::string& combined_file)
        : train_params(combined_file),
          net_params(combined_file) {}

    void Save(const std::string& file)
    {
        train_params.Save(file);
        net_params.Save(file);
    }

    void Load(const std::string& file)
    {
        train_params.Load(file);
        net_params.Load(file);
    }

    void Load(CLI::App& app)
    {
        train_params.Load(app);
        net_params.Load(app);
    }
};
