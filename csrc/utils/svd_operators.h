/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include <torch/torch.h>
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/torch/TorchHelper.h"

using namespace Saiga;
// inline at::Tensor MaskZero(at::Tensor singulars);

class A_function
{
    public:
    virtual ~A_function() = default; // Add a virtual destructor
    virtual torch::Tensor V(const torch::Tensor & vec) = 0;
    virtual torch::Tensor Vt(const torch::Tensor & vec) = 0;
    virtual torch::Tensor U(const torch::Tensor & vec) = 0;
    virtual torch::Tensor Ut(const torch::Tensor & vec) = 0;
    virtual torch::Tensor singulars() = 0;
    virtual torch::Tensor add_zeros(const torch::Tensor & vec) = 0;
    virtual int ratio() = 0;
    void MaskZeroFill(torch::Tensor singulars, torch::Tensor factors);
    torch::Tensor A(const torch::Tensor & vec)
    {
        auto temp = Vt(vec);
        auto s = singulars();
        s = s/s.max();
        return U((s * temp.slice(1, 0, s.size(0)))).reshape({vec.size(0), vec.size(1), vec.size(2) / ratio(), vec.size(3) / ratio()});
    }
    torch::Tensor A_pinv(const torch::Tensor & vec)
    {
        auto vec_shape = vec.sizes();
        auto reshaped_vec = vec.view({vec.size(0),-1});
        auto temp = Ut(reshaped_vec);
        auto singulars_ = singulars();
        singulars_ = singulars_/singulars_.max();
        auto factors = 1.0f/singulars_;
        // auto masks = MaskZero(singulars_).to(torch::kBool);
        // // masks = masks.to(torch::kBool);
        // std::cout << "test here 1 " << TensorInfo(singulars_) << std::endl;
        // factors.index_put_({masks}, torch::tensor(0.0, factors.options()));
        MaskZeroFill(singulars_, factors);
        // factors.index_put_(masks, 0.0); // Handle zero singular values
        // temp.index_put_({at::indexing::Slice(), at::indexing::Slice(0, singulars.size(0))}, 
        //                 temp.index({at::indexing::Slice(), at::indexing::Slice(0, singulars.size(0))}) * factors);
        temp.slice(1,0,singulars_.size(0)) = temp.slice(1,0,singulars_.size(0)) * factors;
        auto out = V(add_zeros(temp)).view({vec_shape[0], vec_shape[1], vec_shape[2] * ratio(), vec_shape[3] * ratio()});
        // std::cout << "input vec " << TensorInfo(vec) << TensorInfo(out) << std::endl;
        return out;

    }

};

inline torch::Tensor mat_by_img(const torch::Tensor & M, const torch::Tensor & v, int dim, int channels)
{
    return torch::matmul(M, v.reshape({v.size(0)*channels, dim, dim}))
            .reshape({v.size(0),channels,M.size(0),dim});
}
inline torch::Tensor img_by_mat(const torch::Tensor &v, const torch::Tensor &M, int dim, int channels)
{
    return torch::matmul(v.reshape({v.size(0)*channels, dim,dim}),M)
            .reshape({v.size(0), channels, dim, M.size(1)});
}

class SuperResolution : public A_function
{
    public:
    SuperResolution(int channels, int img_dim, int ratio, const torch::Device& device)
    : channels(channels), img_dim(img_dim), ratio_(ratio)
    {
        assert(img_dim % ratio == 0); // Ensure img_dim is divisible by ratio
        y_dim = img_dim / ratio;

        // Create matrix A and compute its SVD
        auto A = torch::full({1, ratio * ratio}, 1.0 / (ratio * ratio), torch::TensorOptions().device(device));
        auto svd = torch::svd(A);
        U_small = std::get<0>(svd);
        singulars_small = std::get<1>(svd);
        V_small = std::get<2>(svd);
        Vt_small = V_small.t(); // Transpose of V_small
    }
    // Getter for ratio
    int ratio() override {
        return ratio_;
    }
    // V transformation
    torch::Tensor V(const torch::Tensor& vec) override {
        auto temp = vec.clone().reshape({vec.size(0), -1});
        auto patches = torch::zeros({vec.size(0), channels, y_dim * y_dim, ratio_ * ratio_}, temp.options());

        // Fill patches
        patches.index({"...", 0}) = temp.index({"...", at::indexing::Slice(0, channels * y_dim * y_dim)})
                                       .view({vec.size(0), channels, y_dim * y_dim});
        for (int idx = 0; idx < ratio_ * ratio_ - 1; ++idx) {
            patches.index({"...", idx + 1}) =
                temp.index({"...", at::indexing::Slice(channels * y_dim * y_dim + idx, -1, ratio_ * ratio_ - 1)})
                    .view({vec.size(0), channels, y_dim * y_dim});
        }

        // Multiply each patch by V_small
        patches = torch::matmul(V_small, patches.view({-1, ratio_ * ratio_, 1})).view({vec.size(0), channels, -1, ratio_ * ratio_});

        // Reconstruct the image
        auto patches_orig = patches.view({vec.size(0), channels, y_dim, y_dim, ratio_, ratio_});
        auto recon = patches_orig.permute({0, 1, 2, 4, 3, 5}).contiguous();
        return recon.view({vec.size(0), channels * img_dim * img_dim});
    }

    // V^T transformation
    torch::Tensor Vt(const torch::Tensor& vec) override {
        auto patches = vec.clone().view({vec.size(0), channels, img_dim, img_dim})
                           .unfold(2, ratio_, ratio_)
                           .unfold(3, ratio_, ratio_);
        patches = patches.contiguous().view({vec.size(0), channels, -1, ratio_ * ratio_});

        // Multiply each patch by Vt_small
        patches = torch::matmul(Vt_small, patches.view({-1, ratio_ * ratio_, 1})).view({vec.size(0), channels, -1, ratio_ * ratio_});

        // Reconstruct the vector
        auto recon = torch::zeros({vec.size(0), channels * img_dim * img_dim}, vec.options());
        recon.index({"...", at::indexing::Slice(0, channels * y_dim * y_dim)}) =
            patches.index({"...", 0}).view({vec.size(0), channels * y_dim * y_dim});
        for (int idx = 0; idx < ratio_ * ratio_ - 1; ++idx) {
            recon.index({"...", at::indexing::Slice(channels * y_dim * y_dim + idx, -1, ratio_ * ratio_ - 1)}) =
                patches.index({"...", idx + 1}).view({vec.size(0), channels * y_dim * y_dim});
        }

        return recon;
    }

    // U transformation
    torch::Tensor U(const torch::Tensor& vec) override {
        return U_small.index({0, 0}) * vec.clone().reshape({vec.size(0), -1});
    }

    // U^T transformation
    torch::Tensor Ut(const torch::Tensor& vec) override {
        return U_small.index({0, 0}) * vec.clone().reshape({vec.size(0), -1});
    }

    // Singular values
    torch::Tensor singulars() override {
        return singulars_small.repeat({channels * y_dim * y_dim});
    }

    // Add zeros
    torch::Tensor add_zeros(const torch::Tensor& vec) override {
        auto reshaped = vec.clone().reshape({vec.size(0), -1});
        auto temp = torch::zeros({vec.size(0), reshaped.size(1) * ratio_ * ratio_}, vec.options());
        temp.index({"...", at::indexing::Slice(0, reshaped.size(1))}) = reshaped;
        return temp;
    }
    private:
    int img_dim ;
    int channels ;
    int y_dim;
    int ratio_;
    torch::Tensor U_small, singulars_small, V_small, Vt_small;
};

class SRConv:public A_function
{
    public:
    SRConv(const torch::Tensor & kernel, int channels, int img_dim, const torch::Device& device, int stride = 1)
    :img_dim_(img_dim), channels_(channels), ratio_(stride)
    {
        small_dim_ = img_dim/stride;
        auto A_small = torch::zeros({small_dim_, img_dim_}, torch::TensorOptions().device(device));
        for (int i = stride / 2; i < img_dim + stride / 2; i += stride) {
            for (int j = i - kernel.size(0) / 2; j < i + kernel.size(0) / 2; ++j) {
                int j_effective = j;
                if (j_effective < 0) j_effective = -j_effective - 1;
                if (j_effective >= img_dim) j_effective = (img_dim - 1) - (j_effective - img_dim);
                A_small.index_put_({i / stride, j_effective},
                    A_small.index({i / stride, j_effective}) + 
                    // kernel[j - i + kernel.size(0) / 2].item<float>()
                    kernel.data_ptr<float>()[j - i + kernel.size(0) / 2]
                );
            }
        }
        // SVD of the 1D convolution matrix
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd = torch::svd(A_small, /*some=*/false);
        U_small_ = std::get<0>(svd);
        singulars_small_ = std::get<1>(svd);
        V_small_ = std::get<2>(svd);
        const float ZERO = 3e-2;
        singulars_small_ = torch::where(singulars_small_ < ZERO, torch::zeros_like(singulars_small_), singulars_small_);
        singulars_ = torch::matmul(singulars_small_.reshape({small_dim_,1}), singulars_small_.reshape({1,small_dim_}))
                    .reshape({small_dim_*small_dim_});
        std::vector<int64_t> perm_data;
        for (int i = 0; i < small_dim_; ++i) {
            for (int j = 0; j < small_dim_; ++j) {
                perm_data.push_back(img_dim * i + j);
            }
        }
        for (int i = 0; i < small_dim_; ++i) {
            for (int j = small_dim_; j < img_dim; ++j) {
                perm_data.push_back(img_dim * i + j);
            }
        }
        perm_ = torch::tensor(perm_data, torch::TensorOptions().dtype(torch::kLong).device(device));
    
    }
    int ratio() override {return ratio_;}
    torch::Tensor V(const torch::Tensor &vec) override
    {
        auto temp = torch::zeros({vec.size(0), img_dim_*img_dim_, channels_}, torch::TensorOptions().device(vec.device()));
        auto reshaped_vec = vec.clone().reshape({vec.size(0), img_dim_ * img_dim_, channels_});
        temp.index_put_(
            {torch::indexing::Slice(), perm_, torch::indexing::Slice()},
            reshaped_vec.index({torch::indexing::Slice(), torch::indexing::Slice(0, perm_.size(0)), torch::indexing::Slice()})
        );

        temp.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(perm_.size(0), torch::indexing::None), torch::indexing::Slice()},
            reshaped_vec.index({torch::indexing::Slice(), torch::indexing::Slice(perm_.size(0), torch::indexing::None), torch::indexing::Slice()})
        );
        temp = temp.permute({0,2,1});
        auto out = mat_by_img(V_small_, temp, img_dim_, channels_);
        out = img_by_mat(out, V_small_.transpose(0,1),img_dim_, channels_).reshape({vec.size(0),-1});
        return out;
    }
    torch::Tensor Vt(const torch::Tensor & vec) override
    {
        auto temp = mat_by_img(V_small_.transpose(0,1), vec.clone(), img_dim_ , channels_);
        temp = img_by_mat(temp, V_small_, img_dim_, channels_).reshape({vec.size(0),channels_,-1});
        temp.index_put_({torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(0,perm_.size(0))},
                temp.index({torch::indexing::Slice(), torch::indexing::Slice(), perm_}));
        temp = temp.permute({0,2,1});
        return temp.reshape({vec.size(0),-1});
    }
    torch::Tensor U(const torch::Tensor& vec) override
    {
        // Invert the permutation
        auto temp = torch::zeros({vec.size(0), small_dim_ * small_dim_, channels_}, vec.options());
        temp.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(0, small_dim_ * small_dim_), torch::indexing::Slice()},
            vec.clone().reshape({vec.size(0), small_dim_ * small_dim_, channels_})
        );
        temp = temp.permute({0, 2, 1});  // Permute dimensions: [batch, channels, small_dim^2]

        // Multiply the image by U from the left and by U^T from the right
        auto out = mat_by_img(U_small_, temp, small_dim_, channels_);
        out = img_by_mat(out, U_small_.transpose(0, 1), small_dim_, channels_).reshape({vec.size(0), -1});

        return out;
    }
    torch::Tensor Ut(const torch::Tensor& vec) override
    {
        // Multiply the image by U^T from the left and by U from the right
        auto temp = mat_by_img(U_small_.transpose(0, 1), vec.clone(), small_dim_, channels_);
        temp = img_by_mat(temp, U_small_, small_dim_, channels_)
                .reshape({vec.size(0), channels_, -1});  // Reshape to [batch, channels, small_dim^2]

        // Permute the entries
        temp = temp.permute({0, 2, 1});  // Permute dimensions back to [batch, small_dim^2, channels]
        return temp.reshape({vec.size(0), -1});
    }
    torch::Tensor singulars() override
    {
        return singulars_.repeat_interleave(3).reshape({-1});
    }
    torch::Tensor add_zeros(const torch::Tensor& vec) override
    {
        auto reshaped = vec.clone().reshape({vec.size(0), -1});
        auto temp = torch::zeros(
            {vec.size(0), reshaped.size(1) * ratio_ * ratio_}, vec.options());
        temp.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, reshaped.size(1))}, reshaped);
        return temp;
    }
    private:
    int small_dim_;
    int img_dim_;
    int ratio_;
    int channels_;
    torch::Tensor singulars_, perm_;
    torch::Tensor U_small_, singulars_small_, V_small_;
};

class Deblurring : public A_function
{
  public:
    Deblurring(const torch::Tensor& kernel, int channels, int img_dim, torch::Device device, double ZERO = 3e-2)
        : img_dim_(img_dim), channels_(channels) {
        // Build 1D convolution matrix
        auto A_small = torch::zeros({img_dim_, img_dim_}, torch::TensorOptions().device(device));
        for (int i = 0; i < img_dim_; ++i) {
            for (int j = i - kernel.size(0) / 2; j < i + kernel.size(0) / 2; ++j) {
                if (j < 0 || j >= img_dim_) continue;
                A_small.index_put_({i, j}, kernel[j - i + kernel.size(0) / 2]);
            }
        }

        // SVD of the 1D convolution matrix
        auto svd = torch::svd(A_small, /*some=*/false);
        U_small_ = std::get<0>(svd);
        singulars_small_ = std::get<1>(svd);
        V_small_ = std::get<2>(svd);

        singulars_small_orig_ = singulars_small_.clone();
        singulars_small_ = torch::where(singulars_small_ < ZERO, torch::zeros_like(singulars_small_), singulars_small_);

        // Compute singular values of the big matrix
        singulars_orig_ = torch::matmul(
            singulars_small_orig_.reshape({img_dim_, 1}),
            singulars_small_orig_.reshape({1, img_dim_})
        ).reshape({img_dim_ * img_dim_});

        singulars_ = torch::matmul(
            singulars_small_.reshape({img_dim_, 1}),
            singulars_small_.reshape({1, img_dim_})
        ).reshape({img_dim_ * img_dim_});

        // Sort the singular values and save the permutation
        std::tie(singulars_, perm_) = singulars_.sort(/*descending=*/true);
        singulars_orig_ = singulars_orig_.index({perm_});
    }

    int ratio() override {
        return 1;
    }

    torch::Tensor V(const torch::Tensor& vec) override
    {
        auto temp = torch::zeros({vec.size(0), img_dim_ * img_dim_, channels_}, vec.options());
        temp.index_put_({torch::indexing::Slice(), perm_, torch::indexing::Slice()},
                        vec.clone().reshape({vec.size(0), img_dim_ * img_dim_, channels_}));
        temp = temp.permute({0, 2, 1});
        auto out = mat_by_img(V_small_, temp,img_dim_, channels_);
        out = img_by_mat(out, V_small_.transpose(0, 1), img_dim_, channels_).reshape({vec.size(0), -1});
        return out;
    }

    torch::Tensor Vt(const torch::Tensor& vec) override
    {
        auto temp = mat_by_img(V_small_.transpose(0, 1), vec.clone(),img_dim_, channels_);
        temp = img_by_mat(temp, V_small_,img_dim_,  channels_).reshape({vec.size(0), channels_, -1});
        temp = temp.index({torch::indexing::Slice(), torch::indexing::Slice(), perm_})
                   .permute({0, 2, 1});
        return temp.reshape({vec.size(0), -1});
    }

    torch::Tensor U(const torch::Tensor& vec) override
    {
        auto temp = torch::zeros({vec.size(0), img_dim_ * img_dim_, channels_}, vec.options());
        temp.index_put_({torch::indexing::Slice(), perm_, torch::indexing::Slice()},
                        vec.clone().reshape({vec.size(0), img_dim_ * img_dim_, channels_}));
        temp = temp.permute({0, 2, 1});
        auto out = mat_by_img(U_small_, temp,img_dim_, channels_);
        out = img_by_mat(out, U_small_.transpose(0, 1), img_dim_,  channels_).reshape({vec.size(0), -1});
        return out;
    }

    torch::Tensor Ut(const torch::Tensor& vec) override
    {
        auto temp = mat_by_img(U_small_.transpose(0, 1), vec.clone(),img_dim_, channels_);
        temp = img_by_mat(temp, U_small_, img_dim_,  channels_).reshape({vec.size(0), channels_, -1});
        temp = temp.index({torch::indexing::Slice(), torch::indexing::Slice(), perm_})
                   .permute({0, 2, 1});
        return temp.reshape({vec.size(0), -1});
    }

    torch::Tensor singulars() override
    {
        return singulars_.repeat_interleave(3).reshape({-1});
    }

    torch::Tensor add_zeros(const torch::Tensor& vec) override
    {
        return vec.clone().reshape({vec.size(0), -1});
    }
    private:
    int img_dim_;
    int channels_;
    torch::Tensor U_small_, singulars_small_, V_small_;
    torch::Tensor singulars_small_orig_, singulars_orig_, singulars_, perm_;
};

class Inpainting : public A_function
{
    public:
        Inpainting(int channels__, int img_dim__, const torch::Tensor& missing_indices__, torch::Device device)
            : channels_(channels__), img_dim_(img_dim__) {
            singulars_ = torch::ones({channels_ * img_dim_ * img_dim_ - missing_indices__.size(0)}, torch::TensorOptions().device(device));
            missing_indices_ = missing_indices__;
            std::vector<int64_t> kept_indices__vec;
            for (int i = 0; i < channels_ * img_dim_ * img_dim_; ++i) {
                if (std::find(missing_indices_.data_ptr<int64_t>(), 
                            missing_indices_.data_ptr<int64_t>() + missing_indices_.size(0), 
                            i) == missing_indices_.data_ptr<int64_t>() + missing_indices_.size(0)) {
                    kept_indices__vec.push_back(i);
                }
            }
            kept_indices_ = torch::tensor(kept_indices__vec, torch::TensorOptions().dtype(torch::kLong).device(device));
        }

        int ratio() override {
            return 1;
        }

        torch::Tensor A(const torch::Tensor& vec) 
        {
            vec_shape_ = {vec.size(0), vec.size(1), vec.size(2)}; // Store shape of the input
            auto temp = Vt(vec);
            auto singulars = this->singulars();
            singulars = singulars / singulars.max(); // Normalize singular values
            return U(singulars * temp.index({torch::indexing::Slice(), torch::indexing::Slice(0, singulars.size(0))}));
        }

        torch::Tensor A_pinv(const torch::Tensor& vec) {
            auto reshaped_vec = vec.clone().reshape({vec.size(0), -1});
            auto temp = Ut(reshaped_vec);
            auto singulars = this->singulars();
            singulars = singulars / singulars.max(); // Normalize singular values

            auto factors = 1.0 / singulars;
            factors.index_put_({singulars == 0}, 0.0);
            temp.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, singulars.size(0))}, 
                            temp.index({torch::indexing::Slice(), torch::indexing::Slice(0, singulars.size(0))}) * factors);
            return V(add_zeros(temp)).reshape(vec_shape_);
        }

        torch::Tensor V(const torch::Tensor& vec) override{
            auto temp = vec.clone().reshape({vec.size(0), -1});
            auto out = torch::zeros_like(temp);
            out.index_put_({torch::indexing::Slice(), kept_indices_}, temp.index({torch::indexing::Slice(), 
                        torch::indexing::Slice(0, kept_indices_.size(0))}));
            out.index_put_({torch::indexing::Slice(), missing_indices_}, temp.index({torch::indexing::Slice(), 
                        torch::indexing::Slice(kept_indices_.size(0), torch::indexing::None)}));
            return out.reshape({vec.size(0), -1, channels_}).permute({0, 2, 1}).reshape({vec.size(0), -1});
        }

        torch::Tensor Vt(const torch::Tensor& vec)override {
            auto temp = vec.clone().reshape({vec.size(0), channels_, -1}).permute({0, 2, 1}).reshape({vec.size(0), -1});
            auto out = torch::zeros_like(temp);
            out.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, kept_indices_.size(0))}, temp.index({torch::indexing::Slice(), kept_indices_}));
            out.index_put_({torch::indexing::Slice(), torch::indexing::Slice(kept_indices_.size(0), torch::indexing::None)}, temp.index({torch::indexing::Slice(), missing_indices_}));
            return out;
        }

        torch::Tensor U(const torch::Tensor& vec) override{
            return vec.clone().reshape({vec.size(0), -1});
        }

        torch::Tensor Ut(const torch::Tensor& vec) override{
            return vec.clone().reshape({vec.size(0), -1});
        }

        torch::Tensor singulars() override {
            return singulars_;
        }

        torch::Tensor add_zeros(const torch::Tensor& vec) override{
            auto temp = torch::zeros({vec.size(0), channels_ * img_dim_ * img_dim_}, vec.options());
            auto reshaped = vec.clone().reshape({vec.size(0), -1});
            temp.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, reshaped.size(1))}, reshaped);
            return temp;
        }
    private:
    int channels_;
    int img_dim_;
    torch::Tensor singulars_;
    torch::Tensor missing_indices_;
    torch::Tensor kept_indices_;
    std::vector<int64_t> vec_shape_;
};
inline float bicubic_kernel(float x, float a = -0.5) {
    x = std::abs(x);
    if (x <= 1) {
        return (a + 2) * std::pow(x, 3) - (a + 3) * std::pow(x, 2) + 1;
    } else if (x > 1 && x < 2) {
        return a * std::pow(x, 3) - 5 * a * std::pow(x, 2) + 8 * a * x - 4 * a;
    } else {
        return 0;
    }
}