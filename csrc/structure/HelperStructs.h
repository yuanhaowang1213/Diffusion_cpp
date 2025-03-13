/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/vision/torch/EigenTensor.h"
#include "saiga/vision/torch/ImageTensor.h"

#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include "saiga/core/math/random.h"
#include <torch/nn/parallel/data_parallel.h>

#include <torch/torch.h>
using namespace Saiga;

struct SliceList
{
    // The data from which this slice come from
    torch::Tensor sequence;

    // The target slice 
    torch::Tensor slice;

    // int index;
    torch::Tensor index;

    // The target mask 
    torch::Tensor mask;

    // noise vector
    torch::Tensor noise;
    // std::vector<torch::Tensor> sequences;
    // std::vector<torch::Tensor> slices;
    SliceList() {}

    SliceList(const std::vector<SliceList> & list)
    {
        std::vector<torch::Tensor> sq_list;
        std::vector<torch::Tensor> sl_list;
        std::vector<torch::Tensor> ind_list;
        std::vector<torch::Tensor> sm_list;
        std::vector<torch::Tensor> ni_list;
        for(auto & l:list)
        {
            sq_list.push_back(l.sequence);
            sl_list.push_back(l.slice);
            if(l.index.defined())
            ind_list.push_back(l.index);
            if(l.mask.defined())
            sm_list.push_back(l.mask);
            if(l.noise.defined())
            ni_list.push_back(l.noise);
        }

        sequence = torch::cat(sq_list, 0);
        slice   = torch::cat(sl_list, 0);
        if(ind_list.size()>0)
        index   = torch::cat(ind_list,0);
        if(sm_list.size()>0)
        mask = torch::cat(sm_list, 0);
        if(ni_list.size()>0)
        noise = torch::cat(ni_list, 0);
    }

    void to(torch::Device curr_device)
    {
        sequence = sequence.to(curr_device);
        slice   = slice.to(curr_device);
        if(index.defined())
        index   = index.to(curr_device);
        if(mask.defined())
        mask    = mask.to(curr_device);
        if(noise.defined())
        noise   = noise.to(curr_device);
    }
    // void to(std::vector<torch::Device> curr_devices)
    // {
    //     if(curr_devices.size() == 1)
    //     {
    //         sequence = sequence.to(curr_devices.front());
    //         slice   = slice.to(curr_devices.front());
    //     }
    //     else if (curr_devices.size() > 1)
    //     {
    //         torch::autograd::Scatter scatter(curr_devices, torch::nullopt, 0);
    // // auto scattered_inputs = torch::fmap<torch::Tensor>(scatter.apply({std::move(input)}));
    //         sequences = torch::fmap<torch::Tensor>(scatter.apply({std::move(sequence)}));
    //         slices = torch::fmap<torch::Tensor>(scatter.apply({std::move(slice)}));

    //     }

    // }
};


struct ImageList
{
    // The data from which this slice come from

    // The target slice 
    torch::Tensor image;

    // int index;
    torch::Tensor index;

    std::string filename;
    std::vector<std::string> filenames;
    // std::vector<torch::Tensor> sequences;
    // std::vector<torch::Tensor> slices;
    ImageList() {}

    ImageList(const std::vector<ImageList> & list)
    {
        std::vector<torch::Tensor> sl_list;
        std::vector<torch::Tensor> ind_list;
        std::vector<std::string> fn_list;
        for(auto & l:list)
        {
            sl_list.push_back(l.image);
            if(l.index.defined())
            ind_list.push_back(l.index);
            if(l.filename.empty())
            fn_list.push_back(l.filename);
        }

        image   = torch::cat(sl_list, 0);
        if(ind_list.size()>0)
        index   = torch::cat(ind_list,0);
        filenames = fn_list;
    }

    void to(torch::Device curr_device)
    {
        image   = image.to(curr_device);
        if(index.defined())
        index   = index.to(curr_device);

    }
};