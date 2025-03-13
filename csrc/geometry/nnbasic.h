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
#include <numeric>

#include <typeinfo>
#include <cstdlib>


using namespace Saiga;




inline torch::Tensor get_timestep_embedding(torch::Tensor timesteps, int embedding_dim, float max_period = 10000.0) {
    // Ensure timesteps is a 1D tensor
    TORCH_CHECK(timesteps.dim() == 1, "Timesteps should be a 1D tensor");

    // Compute half dimension
    int half_dim = embedding_dim / 2;
    float emb_factor = std::log(max_period) / (half_dim - 1);

    // Create sinusoidal frequency tensor
    torch::Tensor emb = torch::arange(half_dim, torch::dtype(torch::kFloat32)) * (-emb_factor);
    emb = torch::exp(emb);

    // Ensure emb is on the same device as timesteps
    emb = emb.to(timesteps.device());

    // Compute embedding matrix
    torch::Tensor emb_matrix = timesteps.to(torch::kFloat32).unsqueeze(1) * emb.unsqueeze(0);

    // Concatenate sine and cosine components
    torch::Tensor emb_out = torch::cat({torch::sin(emb_matrix), torch::cos(emb_matrix)}, 1);

    // Zero pad if embedding_dim is odd
    if (embedding_dim % 2 == 1) {
        emb_out = torch::nn::functional::pad(emb_out, torch::nn::functional::PadFuncOptions({0, 1, 0, 0}));
    }

    return emb_out;
}


class TimestepBlock : public torch::nn::Module{
    public:
    virtual torch::Tensor forward(torch::Tensor x, torch::Tensor emb) = 0;
};

class TimestepEmbedSequential : public torch::nn::Module {
public:
    torch::nn::ModuleList layers;  // Store child layers

    // Constructor: Accepts a list of layers
    TimestepEmbedSequential(std::vector<std::shared_ptr<torch::nn::Module>> module_list) {
        for (auto& module : module_list) {
            layers->push_back(module);
        }
        register_module("layers", layers);
    }
    // Alternative constructor for direct module registration
    template<typename... Modules>
    TimestepEmbedSequential(Modules&&... modules) : 
        layers(std::forward<Modules>(modules)...) {
        register_module("layers", layers);
    }
    // Forward function
    torch::Tensor forward(torch::Tensor x, torch::Tensor emb) {
        for (const auto& module : *layers) {
            if (auto timestep_block = std::dynamic_pointer_cast<TimestepBlock>(module)) {
                x = timestep_block->forward(x, emb);  // Pass `emb` if supported
            } else if (auto mod = std::dynamic_pointer_cast<torch::nn::Module>(module)) {
                x = mod->as<torch::nn::AnyModule>()->forward(x);  // Cast and forward
            }
        }
        return x;
    }
};


inline torch::nn::GroupNorm Normalize(int in_channels) {
    return torch::nn::GroupNorm(torch::nn::GroupNormOptions(in_channels / 4, in_channels).eps(1e-6).affine(true));
}

template <typename ModuleType>
ModuleType zero_module(ModuleType module) {
    auto ptr = module.ptr();  // Extract the contained module
    for (auto& p : ptr->parameters()) {
        p.detach_().zero_();
    }
    return module;
}


inline torch::nn::AnyModule conv_nd(
    int dims, 
    int in_channels, 
    int out_channels, 
    int kernel_size, 
    int stride = 1, 
    int padding = 0, 
    bool bias = true) {  

    if (dims == 1) {
        return torch::nn::AnyModule(torch::nn::Conv1d(
            torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .bias(bias)
        ));
    } else if (dims == 2) {
        return torch::nn::AnyModule(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .bias(bias)
        ));
    } else if (dims == 3) {
        return torch::nn::AnyModule(torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .bias(bias)
        ));
    } else {
        throw std::invalid_argument("Unsupported dimensions: " + std::to_string(dims));
    }
}


// Custom SiLU activation function
struct SiLU : torch::nn::Module {
    torch::Tensor forward(torch::Tensor & x) {
        return x * torch::sigmoid(x);
    }
};


class QKVAttention_2LegacyImpl : public torch::nn::Module
{
    
    // A module which performs QKV attention. Matches legacy QKVAttention_2 + input/ouput heads shaping
    public:
    QKVAttention_2LegacyImpl(int num_heads) : num_heads(num_heads) {};

    torch::Tensor forward(const torch::Tensor & qkv)
        // Apply QKV attention.
        // :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        // :return: an [N x (H * C) x T] tensor after attention.
    {
        int bs = qkv.size(0);
        int width = qkv.size(1);
        int length = qkv.size(2);

        // log_ofs << "test qkva " << std::endl;
        // PrintTensorInfo(qkv);

        CHECK_EQ(width % (3 * num_heads), 0);
        int ch = width/(3 * num_heads);
        auto qkv_vec = qkv.reshape({bs * num_heads, ch * 3, length}).split(ch, 1);
        float scale = 1/std::sqrt(std::sqrt(ch));

        auto weight = torch::einsum("bct,bcs->bts", {qkv_vec[0] * scale, qkv_vec[1]*scale});
        weight = torch::softmax(weight.to(torch::kFloat), -1).to(weight.dtype());

        auto a = torch::einsum("bts,bcs->bct", {weight, qkv_vec[2]});
        return a.reshape({bs,-1,length});

    }

    int num_heads;

};
TORCH_MODULE(QKVAttention_2Legacy);

class QKVAttention_2Impl : public torch::nn::Module
{
    public:
    QKVAttention_2Impl(int num_heads) : num_heads(num_heads) {};

    public:
    torch::Tensor forward(torch::Tensor qkv)
    {
        // Apply QKV attention 
        // qkv : an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs 
        // return an [N x (H *C) x T] tensor after attention
        int bs = qkv.size(0);
        int width = qkv.size(1);
        int length = qkv.size(2);

        CHECK_EQ(width % (3 * num_heads), 0);
        int ch = width/ (3 * num_heads);
        auto qkv_vec = qkv.chunk(3,1);
        float scale = 1/std::sqrt(std::sqrt(ch));
        auto weight = torch::einsum("bct,bcs->bts", {qkv_vec[0]*scale, qkv_vec[1]*scale});
        // TODO check function einsum 
        weight = torch::softmax(weight.to(torch::kFloat),-1).to(weight.dtype());
        auto a = torch::einsum("bts,bcs->bct", {weight, qkv_vec[2]});
        return a.reshape({bs,-1,length});
         
    }    

    public:
    int num_heads;
};
TORCH_MODULE(QKVAttention_2);


class Upsample_2Impl : public torch::nn::Module
{
    public:
    Upsample_2Impl(int channels, int out_channels, bool use_conv, int dims =2 )
    :channels(channels),use_conv(use_conv)
    {
        double scale_factor = 2;
        // seq->push_back(torch::nn::Upsample_2)
        bool align_corners = true;
        if(use_conv)
        {
            auto conv = conv_nd(dims, channels, out_channels, 3, 1, 1);
            seq->push_back(conv);
            register_module("seq",seq);

        }
        

    }

    torch::Tensor forward(torch::Tensor x)
    {
        CHECK_EQ(x.size(1), channels);
        // x = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions().scale_factor({2.0, 2.0}).mode(torch::kNearest));
        x = torch::nn::functional::interpolate(
        x, 
        torch::nn::functional::InterpolateFuncOptions()
            .scale_factor(c10::optional<std::vector<double>>{std::vector<double>{2.0, 2.0}})  // ✅ Fix: Explicit std::vector
            .mode(torch::kNearest));

        if(use_conv)
        {
            x = seq->forward(x);
        }
        return x;
        // return seq->forward(x);
    }
    bool use_conv;
    int channels; 
    // std::shared_ptr<torch::nn::Module> conv;  // ✅ Use std::shared_ptr instead of torch::nn::AnyModule
    torch::nn::Sequential seq;
};
TORCH_MODULE(Upsample_2);


class Downsample_2Impl : public torch::nn::Module
{
    public:
    Downsample_2Impl(int channels, int out_channels, bool use_conv, int dims =2)
    : channels(channels)
    {
        if (use_conv) {
            seq->push_back(conv_nd(dims, channels, out_channels, 3, 2, 0));
        } else {
            seq->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2)));
        }
        register_module("seq",seq);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        CHECK_EQ(x.size(1), channels);
        return seq->forward(x);
    }
    int channels;
    torch::nn::Sequential seq;
};
TORCH_MODULE(Downsample_2);

class ResBlock_2Impl :  public torch::nn::Module
{
    public:
    // virtual ~ResBlock_2Impl() = default;
    ResBlock_2Impl(int channels, int out_channels, float dropout, int emb_channels = -1, int dims = 2, 
                bool use_conv = false, bool use_scale_shift_norm = false, bool up = false, bool down = false)
                :emb_channels(emb_channels), use_scale_shift_norm(use_scale_shift_norm)

    {
        in_layers_rest->push_back(Normalize( channels)); 
        /// GroupNorm model(GroupNormOptions(2, 2).eps(2e-5).affine(false));
        //  GroupNormOptions(int64_t num_groups, int64_t num_channels);
        in_layers_rest->push_back(Saiga::ActivationFromString("silu"));

        in_layers_conv->push_back(conv_nd(dims, channels, out_channels, 3, 1, 1));

        register_module("in_layers_rest", in_layers_rest);
        register_module("in_layers_conv", in_layers_conv);
        if(up)
        {
            h_upd->push_back(Upsample_2(channels, channels, false, dims));
            x_upd->push_back(Upsample_2(channels, channels, false, dims));
        }
        else if(down)
        {
            h_upd->push_back(Downsample_2(channels, channels, false, dims));
            x_upd->push_back(Downsample_2(channels, channels, false, dims));
        }
        else
        {
            h_upd->push_back(torch::nn::Identity());
            x_upd->push_back(torch::nn::Identity());
        }
        register_module("h_upd", h_upd);
        register_module("x_upd", x_upd);

        auto make_lin = [](int in, int out)
        {
            auto lin = torch::nn::Linear(in, out);
            torch::nn::init::kaiming_normal_(lin->weight, 0, torch::kFanIn, torch::kReLU);
            return lin;
        };

        if(emb_channels > 0)
        {
            emb_layers->push_back(Saiga::ActivationFromString("silu"));
            if (use_scale_shift_norm)
            {
                emb_layers->push_back(make_lin(emb_channels, 2 * out_channels));
            }
            else
            {
                emb_layers->push_back(make_lin(emb_channels, out_channels));
            }
            register_module("emb_layers", emb_layers);

        }


        out_layers_norm->push_back(Normalize( out_channels));
        out_layers_rest->push_back(Saiga::ActivationFromString("silu"));
        out_layers_rest->push_back(torch::nn::Dropout(dropout));
        out_layers_rest->push_back(zero_module(
                conv_nd(dims, out_channels, out_channels, 3, 1, 1)
            ));

        register_module("out_layers_norm", out_layers_norm);
        register_module("out_layers_rest", out_layers_rest);

        if (out_channels == channels)
        {
            skip_connection->push_back(torch::nn::Identity());
        }
        else if(use_conv)
        {
            skip_connection->push_back(conv_nd(dims, channels, out_channels, 3, 1, 1));
        }
        else
        {
            skip_connection->push_back(conv_nd(dims, channels, out_channels, 1));
        }


        register_module("skip_connection", skip_connection);
        updown = up || down;

    }

    void print_model(const std::string file_name)
    {
        std::ofstream strm;
        if(std::filesystem::exists(file_name))
        {
            strm.open(file_name, std::ios_base::app);
        }
        else
        {
            strm.open(file_name);
        }
        strm << "=================" ;
        strm << "Res Block " ;
        strm << "=================" << std::endl;

        strm << "in_layers_rest " << std::endl;
        strm << in_layers_rest << std::endl; 

        strm << "in_layers_conv " << std::endl;
        strm << in_layers_conv << std::endl;

        strm << "h_upd" << std::endl;
        strm << h_upd << std::endl;

        strm << "x_upd " << std::endl;
        strm << x_upd << std::endl;

        if(emb_channels>0)
        {
            strm << "emb_layers" << std::endl;
            strm << emb_layers << std::endl; 
        }

        // for(auto &p : emb_layers->named_parameters())
        // {
        //     strm << p.key() << std::endl;
        // }
        strm << "out_layers_norm " << std::endl;
        strm << out_layers_norm << std::endl;
        // for(auto &p : out_layers_norm->named_parameters())
        // {
        //     strm << p.key() << std::endl;
        // }
        strm << "out_layers_rest " << std::endl;
        strm << out_layers_rest << std::endl;
        // for(auto &p : out_layers_rest->named_parameters())
        // {
        //     strm << p.key() << std::endl;
        // } 
        strm << "skip_connection " << std::endl;
        strm << skip_connection << std::endl;
        // for(auto &p : skip_connection->named_parameters())
        // {
        //     strm << p.key() << std::endl;
        // }

    }
    torch::Tensor forward(torch::Tensor x, torch::Tensor emb  = {})
    {
        torch::Tensor h;
        if (updown)
        {
            h = in_layers_rest->forward(x);
            h = h_upd->forward(h);
            x = x_upd->forward(x);
            h = in_layers_conv->forward(h);
        }
        else
        {
            h = in_layers_rest->forward(x);
            h = in_layers_conv->forward(h);
        }

        if(emb_channels > 0)
        {
            auto emb_out = emb_layers->forward(emb);
            // log_ofs << "emb_out size " << emb_out.dim() << std::endl;
            while(emb_out.dim() < h.dim())
            emb_out = emb_out.unsqueeze(-1);
            // PrintTensorInfo(emb_out);
            // log_ofs << "updown " << updown << "use_scale_shift_norm " <<  use_scale_shift_norm <<
            // "updown" << updown << std::endl;
            if(use_scale_shift_norm)
            {
                auto scale_shift = emb_out.chunk(2, 1);

                // PrintTensorInfo(scale_shift[0]);
                // PrintTensorInfo(scale_shift[1]);
                h = out_layers_norm->forward(h) * (1 + scale_shift[0]) + scale_shift[1];
                h = out_layers_rest->forward(h);
            }
            else
            {
                h = h + emb_out;
                h = out_layers_norm->forward(h);
                h = out_layers_rest->forward(h);
            }
        }
        else
        {
            h = out_layers_norm->forward(h);
            h = out_layers_rest->forward(h);
        }
        // log_ofs << "conduct resblock " << std::endl;
        return skip_connection->forward(x) + h;
    }

    torch::nn::Sequential in_layers_rest, in_layers_conv;
    torch::nn::Sequential h_upd;
    torch::nn::Sequential x_upd;
    torch::nn::Sequential emb_layers;
    torch::nn::Sequential out_layers_norm, out_layers_rest;
    torch::nn::Sequential skip_connection;

    bool updown;
    bool use_scale_shift_norm;
    int emb_channels;

};
TORCH_MODULE(ResBlock_2);

class AttentionBlock_2Impl : public torch::nn::Module
{
    public:
    AttentionBlock_2Impl( int channels, int num_heads_in =1, int num_head_channels=-1, bool use_new_attention_order = false)
    {
        // std::cout << "channels " << channels << " " << num_heads_in << " " << num_head_channels << std::endl;
        int num_heads;
        if(num_head_channels == -1)
        {
            num_heads = num_heads_in;
        }
        else
        {
            CHECK_EQ(channels % num_head_channels, 0) << "q,k,v channels " << channels <<" is not divieded by num_head_channels "<< num_head_channels << std::endl;
            num_heads = channels/num_head_channels;
        }

        seq->push_back(Normalize( channels));
        // seq->push_back(torch::nn::Conv1d{torch::nn::Conv1dOptions(channels, 3*channels,1).padding(0).padding_mode(torch::kCircular)});
        seq->push_back(conv_nd(1, channels, channels * 3, 1));
        if(use_new_attention_order)
        {
            seq->push_back(QKVAttention_2(num_heads));
        }
        else
        {
            seq->push_back(QKVAttention_2Legacy(num_heads));
        }
        

        seq->push_back(zero_module(
                conv_nd(1, channels, channels,  1)
            ));

        register_module("seq", seq);
    }

    void print_model(const std::string file_name)
    {
        std::ofstream strm;
        if(std::filesystem::exists(file_name))
        {
            strm.open(file_name, std::ios::app);
        }
        else
        {
            strm.open(file_name);
        }
        strm << "=================" ;
        strm << "AttentionBlock_2 " ;
        strm << "=================" << std::endl;
        strm << seq << std::endl;
        // for(auto &p : seq->named_parameters())
        // {
        //     strm << p.key() << std::endl;
        // }
    }
    torch::Tensor forward(torch::Tensor x)
    {

        auto x_size = x.sizes();
        x = x.reshape({x_size[0],x_size[1],-1});
        auto h = seq->forward(x);

        h = (x+h).reshape(x_size);

        return h;
    }
    torch::nn::Sequential seq;
};
TORCH_MODULE(AttentionBlock_2);


class EDM_UNETModelImpl : public torch::nn::Module
{
    // This UNet model with attention and timestep embedding 
    // param in_channels: channels in the input Tensor 
    // param model_channels: base channel count for the model 
    // param num_res_blocks number residual blocks per downsample 
    // param attention_resolutions: a collection of downsample rates at which attention will take place 
    // Maybe a set, list or tuple 
    // For example, if this contains 4, then at 4x downsampling, attention will be used 
    // param droupout: the dropout probability 
    // param channel_mult: channel multiplier for each level of the UNet 
    // param conv_resample: if True, use learned convolutions for upsampling and downsampling 
    // param dims : determins if the signal is 1D, 2D or 3D
    // param num_class: if specified (as an int), then this model will be class-conditional with 'num_classes' classes 
    // param use_checkpoint: use gradient checkpointing to reduce memeory usage.
    // param num_heads: the number of attention heads in each attention layer. 
    // param num_head_channels : if specified, ignore num_heads and instead use a fixed channel width per attention head 
    // param num_heads_upsample : works with num_heads to set a differetnt number of heads for upsampling. Deprecated 
    // param use_scale_shift_norm: use residual blocks for up/downsampling
    // param resblock_updown: use residual blocks for up/downsampling 
    // param use_new_attention_order: use a different attetntion pattern for potentially increased efficiency 
    public: EDM_UNETModelImpl(int in_channels, int out_channels, 
    std::shared_ptr<CombinedParams> params, bool conv_resample= true, int dims =2, 
    std::string pool = "adaptive")
    : in_channels(in_channels), out_channels(out_channels), params(params), conv_resample(conv_resample), dims (dims), pool(pool)
    {
        // parameter preprocessing 
        std::string model_filename = params->train_params.experiment_dir + "/model.txt";;
        if(params->train_params.validate_on)
        {
            model_filename = params->train_params.experiment_dir + "/model_validate.txt";;
        }

        // std::cout << "attn resolution " << std::atoi(params->train_params.attention_resolution[0].c_str())  << std::endl;
        // // std::cout << "attn " << static_cast<int>(params->train_params.attention_resolution[0].c_str()) << std::endl;
        // std::cout << "attn res " << params->train_params.attention_resolution[0] << std::endl;
        std::vector<int> attention_resolution; 

        for (const auto& attn_res_str : params->train_params.attention_resolution) {
            int attention_res = params->train_params.max_dim / std::stoi(attn_res_str);
            attention_resolution.push_back(attention_res);
        }
        log_ofs << "Attention resolution is " << attention_resolution << std::endl;
        std::vector<int> channel_mult;
        if(params->train_params.channel_mult.size() > 0)
        {
            for(int i = 0; i < params->train_params.channel_mult.size(); ++i)
            {
                channel_mult.push_back(stoi(params->train_params.channel_mult[i]));
            }
        }

        int num_heads_upsample;
        if(params->train_params.num_heads_upsample == -1)
        {
            num_heads_upsample= params->train_params.num_heads;
        }

        model_channels = params->train_params.model_channels;
        int time_embed_dim = params->train_params.model_channels * 4;
        num_classes = params->train_params.num_classes;

        // model description 
        auto make_lin = [](int in, int out)
        {
            auto lin = torch::nn::Linear(in, out);
            torch::nn::init::kaiming_normal_(lin->weight, 0, torch::kFanIn, torch::kReLU);
            return lin;
        };
        time_embed->push_back(make_lin(params->train_params.model_channels, time_embed_dim));
        time_embed->push_back(Saiga::ActivationFromString("silu"));
        time_embed->push_back(make_lin(time_embed_dim, time_embed_dim));
        register_module("time_embed", time_embed);


        if(num_classes > 0)
        {
            label_emb->push_back(torch::nn::Embedding{torch::nn::EmbeddingOptions(num_classes, time_embed_dim)});
            register_module("label_emb", label_emb);
        }

        int ch = channel_mult[0] * params->train_params.model_channels;
        int input_ch = ch;

        int feature_size_ = ch;
        // auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, ch, 3)
        //     .stride(1)
        //     .padding(1));
        //     conv_nd(2, ch, out_channels, out_channels, 3, 1, 1)
        // Register the Conv2d module
        // register_module("conv1", conv);
        
        input_blocks->push_back(conv_nd(2,in_channels, ch, 3, 1,1));
        // if(dims == 2)
        // {
        //     auto conv_layer = torch::nn::Conv2d{torch::nn::Conv2dOptions(in_channels, ch,3).padding(1).padding_mode(torch::kCircular)};
        //     input_blocks->push_back(conv_layer);
        // }
        // else
        // {
        //     CHECK(false) << "currently only support 2D " << std::endl;
        // }
        // register_module("input_blocks_pre", input_blocks_pre);
        // std::cout << "resblock up down " <<params->train_params.resblock_updown << std::endl;
        std::vector<int> input_block_chans;
        input_block_chans.push_back(ch);
        int ds = 1;


        for(int i = 0; i < channel_mult.size(); ++i)
        {
            for(int j = 0; j < params->train_params.num_res_blocks; ++j)
            {
                input_blocks->push_back(ResBlock_2(ch, channel_mult[i] * params->train_params.model_channels, 
                                    params->train_params.dropout, time_embed_dim,  dims, false,params->train_params.use_scale_shift_norm ));
                ch = channel_mult[i] * params->train_params.model_channels;
                // if (ds == attention_resolution)
                for(int k = 0; k < attention_resolution.size(); ++k)
                {
                    if(ds == attention_resolution[k])
                    {
                        auto attention_block = AttentionBlock_2(ch, params->train_params.num_heads,params->train_params.num_heads_channels, params->train_params.use_new_attention_order);
                        input_blocks->push_back(attention_block);
                    }
                }
                feature_size_ += ch;
                input_block_chans.push_back(ch);

            }
            if (i != (channel_mult.size() -1))
            {
                int out_ch = ch;
                if(params->train_params.resblock_updown)
                {
                    input_blocks->push_back(ResBlock_2(ch, out_ch,  params->train_params.dropout, time_embed_dim, 
                                                    dims, false, params->train_params.use_scale_shift_norm,false, true ));
                }
                else
                {
                    input_blocks->push_back(Downsample_2(ch, out_ch, conv_resample, dims));
                }
                ch = out_ch;
                input_block_chans.push_back(ch);
                ds *= 2;
                feature_size_ += ch;

            }

        }
        register_module("input_blocks", input_blocks);

        // auto resblock = 
        // resblock_name = resblock.type_info().name();


        middle_blocks->push_back(ResBlock_2(ch,  ch, params->train_params.dropout, time_embed_dim, 
                                dims, false, params->train_params.use_scale_shift_norm));
        
        // auto attention_block = AttentionBlock_2_u(ch,  params->train_params.num_heads,params->train_params.num_heads_channels, params->train_params.use_new_attention_order);
        // attention_block->print_model(model_filename);
        middle_blocks->push_back(AttentionBlock_2(ch,  params->train_params.num_heads,params->train_params.num_heads_channels, params->train_params.use_new_attention_order));

        middle_blocks->push_back(ResBlock_2(ch,  ch, params->train_params.dropout, time_embed_dim,
                                dims, false, params->train_params.use_scale_shift_norm));

        register_module("middle_blocks", middle_blocks);

        attnblock_name =  typeid(*middle_blocks[1]).name();
        // resblock_name = middle_blocks->ptr(0)->as<torch::nn::AnyModule>()->type_info().name();
        // resblock_name = middle_blocks[0]->type_info().name();
        for(auto  module : *middle_blocks)
        {
            // log_ofs << "test 3" << std::endl;
            // auto test = module.forward(x, emb);
            resblock_name = module.type_info().name();
        }

        feature_size_ += ch;

        // for(int i = 0; i < channel_mult.size()-1;++i)

        log_ofs << "input_blocks_chans " << input_block_chans.size() << std::endl;
        log_ofs << "num re blocks " << params->train_params.num_res_blocks << std::endl;
        for(int i = channel_mult.size()-1; i >= 0; --i)
        {
            for(int j = 0; j < params->train_params.num_res_blocks +1; ++j)
            {
                padding_flag.push_back(1);
                int ich = input_block_chans.back();
                input_block_chans.pop_back();
                output_blocks->push_back(ResBlock_2(ch+ich,  params->train_params.model_channels * channel_mult[i],
                             params->train_params.dropout, time_embed_dim, dims, false, params->train_params.use_scale_shift_norm));
                ch = params->train_params.model_channels  * channel_mult[i];
                for(int k = 0; k < attention_resolution.size(); ++k)
                {
                    if(ds == attention_resolution[k])
                    {
                        padding_flag.push_back(-1);
                        output_blocks->push_back(AttentionBlock_2(ch, num_heads_upsample, params->train_params.num_heads_channels, params->train_params.use_new_attention_order));

                    }

                    if( i > 0 && (j == params->train_params.num_res_blocks))
                    {
                        int out_ch = ch;
                        padding_flag.push_back(-1);
                        if(params->train_params.resblock_updown)
                        {
                            output_blocks->push_back(ResBlock_2(ch, out_ch, params->train_params.dropout, time_embed_dim, 
                                            dims, false, params->train_params.use_scale_shift_norm, true, false));
  
                        }
                        else
                        {
                            // Downsample_2(ch, out_ch, conv_resample, dims)
                            output_blocks->push_back(Upsample_2(ch, out_ch, conv_resample, dims));
                        }
                        ds /=2;
                    }
                }
                feature_size_+=ch;

            }

        }
        register_module("output_blocks", output_blocks);
        out->push_back(torch::nn::GroupNorm(32, ch));
        out->push_back(Saiga::ActivationFromString("silu"));
        
        if(dims == 1)
        {
            auto conv_layer = torch::nn::Conv1d{torch::nn::Conv1dOptions(input_ch, out_channels,3).padding(1).padding_mode(torch::kCircular)};
            torch::nn::init::constant_(conv_layer->weight, 0);
            torch::nn::init::constant_(conv_layer->bias, 0);
            // TO DO test 
            // for p in module.parameters():
            //     p.detach().zero_()
            out->push_back(conv_layer);
            
        }
        else if(dims == 2)
        {
            auto conv_layer = torch::nn::Conv2d{torch::nn::Conv2dOptions(input_ch, out_channels,3).padding(1).padding_mode(torch::kCircular)};
            torch::nn::init::constant_(conv_layer->weight, 0);
            torch::nn::init::constant_(conv_layer->bias, 0);
            // TO DO test 
            // for p in module.parameters():
            //     p.detach().zero_()
            out->push_back(conv_layer);
        }
        else if(dims == 3)
        {
            auto conv_layer = torch::nn::Conv3d{torch::nn::Conv3dOptions(input_ch, out_channels,3).padding(1).padding_mode(torch::kCircular)};
            torch::nn::init::constant_(conv_layer->weight, 0);
            torch::nn::init::constant_(conv_layer->bias, 0);
            // TO DO test 
            // for p in module.parameters():
            //     p.detach().zero_()
            out->push_back(conv_layer);
        }
        else
        {
            CHECK(false) << "not support dim in resblock" << std::endl;
        }
        register_module("out",out);
        // for(auto & p : input_blocks->named_parameters())
        // {
        //     log_ofs << p.key() << std::endl;
        //     log_ofs << p.value() << std::endl;
        // }
        log_ofs << "resblock name is " <<resblock_name << std::endl;
        log_ofs << "padding block size " << padding_flag.size() << std::endl;
        log_ofs << padding_flag << std::endl;
        print_model(model_filename);
    }

    void print_model(const std::string file_name)
    {
        std::ofstream strm;
        if(std::filesystem::exists(file_name))
        {
            strm.open(file_name, std::ios::app);
        }
        else
        {
            strm.open(file_name);
        }
        strm << "=================" ;
        strm << "UNet Model " ;
        strm << "=================" << std::endl;

        strm << "time_embed " << std::endl;
        strm << time_embed << std::endl; 
        // for(auto &p : time_embed->named_parameters())
        // {
        //     strm << p.key() << std::endl;
        // }

        strm << "label_emb " << std::endl;
        strm << label_emb << std::endl;
        // for(auto &p : label_emb->named_parameters())
        // {
        //     strm << p.key() << std::endl;
        // }


        strm << "input_blocks " << std::endl;
        strm << input_blocks << std::endl;
        // for(auto &p : input_blocks->named_parameters())
        // {
        //     strm << p.key() << std::endl;
        // }

        strm << "middle_blocks " << std::endl;
        strm << middle_blocks << std::endl;

        strm << "output_blocks " << std::endl;
        strm << output_blocks << std::endl;


        strm << "out " << std::endl;
        strm << out << std::endl;

    }

    torch::Tensor forward(const torch::Tensor & x, torch::Tensor timesteps, torch::Tensor y = torch::Tensor())
    {

        CHECK_EQ(y.defined(), num_classes>0);
        auto emb = time_embed->forward(get_timestep_embedding(timesteps, model_channels));
        if(num_classes > 0)
        {
            CHECK_EQ(y.size(0), x.size(0));
            emb = emb + label_emb->forward(y);
        }
        std::vector<torch::Tensor> hs;
        torch::Tensor h = x;


        for(auto & module : *input_blocks)
        {

            // log_ofs << module.type_info().name() << std::endl;
            if(module.type_info().name() == resblock_name)
            // if(dynamic_cast<TimestepEmbed*>(module.ptr())!=nullptr)
            {
                h = module.forward<torch::Tensor>(h, emb);
                hs.push_back(h);
            }
            else
            // log_ofs << typeid(module) << std::endl;
            {

                h = module.forward<torch::Tensor>(h);
                if(module.type_info().name() != attnblock_name)
                {
                    hs.push_back(h);
                }
            }
        }

        for(auto && module : * middle_blocks)
        {
            if(module.type_info().name() == resblock_name)
            {
                h = module.forward(h, emb);
            }
            else
            {

                h = module.forward(h);
            }
        }
        int outflag_count = 0 ;

        for(auto && module : * output_blocks)
        {
            // log_ofs << module.type_info().name() << std::endl;

            if(padding_flag[outflag_count] > 0)
            {
                h = torch::cat({h, hs.back()}, 1);
                hs.pop_back();
            }

            outflag_count += 1;
            // PrintTensorInfo(h);
            if(module.type_info().name() == resblock_name)
            {
                h = module.forward(h, emb);
            }
            else
            {
                h = module.forward(h);
            }

            // log_ofs << "after module " << std::endl;
            // PrintTensorInfo(h);
        }

        return out->forward(h);  
    }

    torch::nn::Sequential time_embed;
    torch::nn::Sequential label_emb;
    // torch::nn::Sequential input_blocks_pre;
    torch::nn::Sequential input_blocks;
    // torch::nn::ModuleList input_blocks,output_blocks;

    torch::nn::Sequential middle_blocks;
    torch::nn::Sequential output_blocks;
    torch::nn::Sequential out;
    int model_channels;
    int num_classes;
    std::string resblock_name, attnblock_name;
    std::vector<int> padding_flag;


    int in_channels;
    int out_channels; 
    std::shared_ptr<CombinedParams> params;
    bool conv_resample;
    int dims;
    std::string pool;

};
TORCH_MODULE(EDM_UNETModel);


// Magnitude-preserving Fourier features 
class MPFourierImpl : public torch::nn::Module
{
    public:
        MPFourierImpl(int num_channels, float bandwidth =1.0)
        {
            freqs   = 2 * pi<double>() * torch::randn({num_channels}) * bandwidth;
            phases  = 2 * pi<double>() * torch::rand({num_channels});
            register_buffer("freqs", freqs);
            register_buffer("phases", phases); 
        };
        torch::Tensor forward(torch::Tensor x)
        {
            auto y = x.to(torch::kFloat32);
            y = y.outer(freqs);
            // originally using ger alias of outer
            y = y + phases;
            y = y.cos() * std::sqrt(2);

            return y.to(x.dtype());
        }

    torch::Tensor freqs;
    torch::Tensor phases;
};
TORCH_MODULE(MPFourier);

inline torch::Tensor normalize_tensor(torch::Tensor x, std::vector<long int> dim ={}, float eps=1e-4)
{
    if(dim.size() == 0)
    {
        for(int i = 0; i < x.dim()-1;++i)
        {
            dim.push_back(i+1);
        }
    }

    torch::IntArrayRef dim_arr = torch::makeArrayRef<long int>(dim);
    
    torch::Tensor norm = torch::linalg::vector_norm(x,2,dim_arr,true, torch::kFloat32);

    norm = eps + std::sqrt(float(norm.numel())/float(x.numel())) * norm;
    // log_ofs<<"tensor x " << TensorInfo(x) << " norm " << TensorInfo(norm) << " factor " <<std::sqrt(float(norm.numel())/float(x.numel())) << std::endl;
    return x/norm;
}


class MPConvImpl : public torch::nn::Module
{
    public:
        MPConvImpl(int in_channels, int out_channels, std::vector<long int> kernel=std::vector<int64_t>())
        :out_channels(out_channels)
        {
            std::vector<long int> kernel_size;
            kernel_size.push_back(out_channels);
            kernel_size.push_back(in_channels);
            if(kernel.size()>0)
            {
                for(int i = 0; i < kernel.size(); ++i)
                {
                    kernel_size.push_back(kernel[i]);
                }
            }

            torch::IntArrayRef weight_arr = torch::makeArrayRef<long int>(kernel_size);

            weights = torch::randn(weight_arr);
            register_parameter("weights", weights);
            
        }
        // torch::Tensor forward(torch::Tensor x, torch::Tensor gain =torch::Tensor())sh
        torch::Tensor forward(torch::Tensor x, torch::Tensor gain=torch::Tensor() )
        {
            // torch::Tensor gain;
            if(!gain.defined())
            gain = torch::tensor({1}, torch::TensorOptions(x.device()).dtype(torch::kFloat));
            auto w = weights;
            if(weights.requires_grad())
            {
                torch::NoGradGuard ngg;
                weights.copy_(normalize_tensor(w));

            }
            w = normalize_tensor(w);
            w = w *(gain/std::sqrt(float(w.slice(0,0,1).numel())));

            // log_ofs <<"input dim " << TensorInfo(x);
            if(w.dim()==2)
            {
                // return x * w.t();
                return softplus(torch::matmul(x, w.t().contiguous()));
            }
            CHECK_EQ(w.dim(), 4);
            // log_ofs << "x " << TensorInfo(x);
            // log_ofs << "w " << TensorInfo(w);
            x = torch::nn::functional::conv2d(x,w, torch::nn::functional::Conv2dFuncOptions().padding(w.size(-1)/2));
            // log_ofs << "after conv " << TensorInfo(x);
            x = softplus(x);

            return x;
        }
    torch::Tensor weights;
    int out_channels;
    torch::nn::Softplus softplus;
};
TORCH_MODULE(MPConv);

class MPConv2Impl : public torch::nn::Module 
{
    public:
    MPConv2Impl(int in_channels, int out_channels, std::vector<int64_t> kernel=std::vector<int64_t>())
    : out_channels(out_channels)
    {
        std::vector<long int> kernel_size;
        kernel_size.push_back(out_channels);
        kernel_size.push_back(in_channels);
        if(kernel.size()>0)
        {
            for(int i = 0; i < kernel.size(); ++i)
            {
                kernel_size.push_back(kernel[i]);
            }
        }
        torch::IntArrayRef weight_arr = torch::makeArrayRef<long int>(kernel_size);

        weights = torch::randn(weight_arr);
        register_parameter("weights", weights);
    }


    torch::Tensor forward(torch::Tensor x) {

        torch::Tensor gain;
        if(!gain.defined())
        gain = torch::tensor({1}, torch::TensorOptions(x.device()).dtype(torch::kFloat));

        torch::Tensor w = weights.to(torch::kFloat32);

        if (weights.requires_grad()) {
            torch::NoGradGuard no_grad;
            weights.copy_(normalize_tensor(w)); // Forced weight normalization
        }

        w = normalize_tensor(w); // Traditional weight normalization
        w = w * (gain / std::sqrt(static_cast<float>(w[0].numel()))); // Magnitude-preserving scaling


        w = w.to(x.dtype());

        if (w.dim() == 2) {
            auto output = 1.5 * torch::sigmoid(x.matmul(w.t())); // Alternative activation function

            return output;
        }

        assert(w.dim() == 4);
        torch::Tensor out = torch::nn::functional::conv2d(x, w, torch::nn::functional::Conv2dFuncOptions().padding({w.size(-1) / 2}));
        out = softplus(out);
        TORCH_CHECK(torch::all(out >= 0).item<bool>(), "out contains values less than or equal to 0!");
        return out;
    }

    torch::Tensor weights;
    torch::nn::Softplus softplus;
    int out_channels;
};
TORCH_MODULE(MPConv2);


