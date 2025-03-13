/**
 * Copyright (c) 2025 Yuanhao Wang
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/image/freeimage.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/ColorizeTensor.h"
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/ImageTensor.h"
#include "saiga/vision/torch/TrainParameters.h"
#include <chrono>

#include "Settings.h"
#include "data/Dataloader.h"
#include "geometry/all.h"
#include "utils/utils.h"

#include "build_config.h"
#include "tensorboard_logger.h"
#include "utils/svd_operators.h"

#include "log_file.h"

#include <boost/log/core.hpp>
using namespace Saiga;

#ifdef UNUSED
#elif defined(__GNUC__)
# define UNUSED(x) UNUSED_ ## x __attribute__((unused))
#elif defined(__LCLINT__)
# define UNUSED(x) /*@unused@*/ x
#else
# define UNUSED(x) x
#endif



struct TrainScene
{
    std::shared_ptr<ImgBase> scene;
    // Here remove tree
    std::shared_ptr<MriNeuralGeometry> neural_geometry;

    double last_eval_loss       = 999999999999999999;
    double new_eval_loss        = 999999999999999999;
    
    void SaveCheckpoint(const std::string& dir)
    {
        auto prefix = dir + "/" + scene->scene_name + "_";
        scene->SaveCheckpoint(dir);

        neural_geometry->SaveCkpt(dir);

        torch::nn::ModuleHolder<MriNeuralGeometry> holder(neural_geometry);

        torch::save(holder, prefix + "geometry.pth");

    }
    void LoadCheckpoint(const std::string& dir)
    {
        // printf("dir is " );
        log_ofs  << "dir is " << dir << std::endl;
        auto prefix = dir+"/" + scene->scene_name + "_";
        scene->LoadCheckpoint(dir);
        log_ofs  << "Load Checkpoint " << dir << std::endl;
        neural_geometry->LoadCkpt(dir);

    }

    
};


class Trainer
{
    public:
        Trainer(std::shared_ptr<CombinedParams> params, std::string experiment_dir)
        : params(params), experiment_dir(experiment_dir)
        {
            torch::set_num_threads(params->train_params.train_num_threads);
            torch::manual_seed(params->train_params.random_seed);

            
            if(params->train_params.validate_on)
            {
                tblogger = std::make_shared<TensorBoardLogger>((experiment_dir + "/tfevents_validate.pb").c_str());
            }
            else
            {
                tblogger = std::make_shared<TensorBoardLogger>((experiment_dir + "/tfevents.pb").c_str());
            }

            for(auto scene_name: params->train_params.scene_name)
            {
                log_ofs << "Scene name " << scene_name << std::endl;
                auto scene = std::make_shared<ImgBase>(params->train_params.scene_dir + "/" + scene_name, params);
                params->train_params.data_type = scene->dataset_params.data_type;
                scene->params = params;
                

                log_ofs << "Train indices tmp " << (scene->scene_path + "/" + params->train_params.split_name + "/train_index.txt") << std::endl;
                if(std::filesystem::exists(scene->scene_path + "/" + params->train_params.split_name + "/train_index.txt"))
                {
                    temp_train_indices = TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/train_index.txt");
                }
                if(std::filesystem::exists(scene->scene_path + "/" + params->train_params.split_name + "/test_index.txt"))
                {
                    temp_test_indices = TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/test_index.txt");
                }
                log_ofs << "temp train indices " << temp_train_indices.size() << " " <<temp_test_indices.size() << std::endl;

                

                {
                    log_ofs << "scene path " << scene->scene_path << " " << params->train_params.split_name << std::endl;
                    if(params->train_params.validate_on)
                    CHECK(std::filesystem::exists(scene->scene_path + "/" + params->train_params.split_name + "/validate.txt"));
                    {
                        scene->validate_indices = 
                            TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/validate.txt");

                        log_ofs << "Validate (" << scene->validate_indices.size() << "): " << array_to_string(scene->validate_indices, ' ')
                                << std::endl;
                    
                    }

                }
                {
                    log_ofs << "scene path " << scene->scene_path << " " << params->train_params.split_name << std::endl;
                    if(!params->train_params.validate_on)
                    {

                        scene->train_indices = 
                        TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/train.txt");
                        scene->test_indices = 
                        TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/eval.txt");
                        if(params->train_params.train_on_eval)
                        {
                            scene->test_indices = scene->train_indices;
                        }

                        log_ofs << "Train (" << scene->train_indices.size() << "): " << array_to_string(scene->train_indices, ' ')
                                    << std::endl;
                        log_ofs << "Test (" << scene->test_indices.size() << "):" << array_to_string(scene->test_indices, ' ')
                                    << std::endl;

                        log_ofs << "load train indices " << scene->train_indices <<std::endl;
                        log_ofs << "load test indices " << scene->test_indices <<std::endl;
                    }

                }

                log_ofs << "validate_echo " << params->train_params.validate_echo << std::endl;

                TrainScene ts;
                
                ts.neural_geometry = std::make_shared<Geometry_DiffUn> (scene->num_channels, scene->D, params);
                ts.neural_geometry->to(devices.front());

                ts.scene = scene;



                log_ofs << "ckpt_directory size " << params->train_params.checkpoint_directory.size() << std::endl;

                if(params->train_params.checkpoint_directory.size() > 0)
                {
                    log_ofs << "load check point directory " << params->train_params.checkpoint_directory << std::endl;

                    ts.neural_geometry->LoadCkpt(params->train_params.checkpoint_directory);
                }


                scenes.push_back(ts);
                
            }

        }

    private:
        template<class T>
        void Eval_Data(std::unique_ptr<torch::data::StatelessDataLoader<T, torch::data::samplers::SequentialSampler>>* data_loader, 
                TrainScene * ts, int epoch_id, std::string name, std::string saving_dir, int max_sample_slice,  std::vector<int> indices, std::vector<int> volume_size)
        {
            std::string ep_str = Saiga::leadingZeroString(epoch_id, 4);

            std::string checkpoint_name = "Checkpoint(" + ep_str + ")" ;
            

            int total_sample_num = indices.size() ;
            
            auto neural_geometry = ts->neural_geometry;
            neural_geometry->ema_apply_shadow();

            Saiga::ProgressBar bar(std::cout, name + " " + std::to_string(epoch_id) + "|", total_sample_num, 30, false, 1000, "slice");


            std::vector<torch::Tensor> gt_volume;
            std::vector<torch::Tensor> predict_volume;
            std::vector<torch::Tensor> gt_volume_complex;
            std::vector<torch::Tensor> pred_volume_complex;
            std::vector<torch::Tensor> input_volume;

            int slice_shape = 1;

            int scale = 4;
            std::vector<float> k_vec(scale * 4);
            for (int i = 0; i < scale * 4; ++i) {
                float x = (1.0 / scale) * (i - std::floor(scale * 4 / 2.0) + 0.5);
                k_vec[i] = bicubic_kernel(x);
            }

            auto k_tensor = torch::from_blob(k_vec.data(), {static_cast<int64_t>(k_vec.size())}, torch::TensorOptions().dtype(torch::kFloat));
            k_tensor /= k_tensor.sum();
            SRConv A_funcs(k_tensor, 3,256, devices.front(), scale);


            std::function<torch::Tensor(const torch::Tensor &)> A = [&A_funcs](const torch::Tensor &input) {
                return A_funcs.A(input);
            };
            std::function<torch::Tensor(const torch::Tensor &)> Ap = [&A_funcs](const torch::Tensor &input) {
                return A_funcs.A_pinv(input);
            };
            
            auto start = std::chrono::high_resolution_clock::now(); // Start time

            for(ImageData sample_data : (**data_loader))
            {


                torch::Tensor index_id = sample_data.images.index;
                // std::cout << "index_id is " << index_id << std::endl;
                auto slice = sample_data.images.image.to(devices.front());
                auto image = neural_geometry->sampling(A(slice), A, Ap);

                bar.addProgress(sample_data.num_input());
            }
            bar.Quit();
            auto end = std::chrono::high_resolution_clock::now(); // End time
            std::chrono::duration<double> elapsed = end - start;
        
            std::cout << "Execution Time: " << elapsed.count() << " seconds\n";

        }
    public:
        void ValidateStep( )
        {
            std::string name = "/Validate";
            // std::vector<std::vector<int>> indices_list;
            // std::vector<std::shared_ptr<MriBase>> scene_list;

            std::string ep_str = "validate_exp";

            std::string checkpoint_name = "Checkpoint(" + ep_str + ")" ;
            auto ep_dir = experiment_dir + "ep" + ep_str + "/";

            std::filesystem::create_directories(ep_dir);

            torch::NoGradGuard ngg;
            for(auto & ts: scenes)
            {
                auto scene           = ts.scene;
                auto neural_geometry = ts.neural_geometry;


                neural_geometry->train(0, false);
                std::vector<int> indices;

                indices = scene->validate_indices;

                int max_sample_slice = params->train_params.max_dim;
                auto options = torch::data::DataLoaderOptions().batch_size(params->train_params.batch_size).drop_last(false).workers(params->train_params.num_workers_eval);


                std::cout << "tmp test indices " << temp_test_indices << " max dim " << max_sample_slice << std::endl;
                auto sampler = torch::data::samplers::SequentialSampler(indices.size() );
                // auto sampler = torch::data::samplers::SequentialSampler(indices.size() *max_sample_slice);
                auto dataset = SequentialImageDataset(indices, scene,  params);
                auto data_loader = torch::data::make_data_loader(dataset, sampler, options);
                Eval_Data<SequentialImageDataset>
                    (& data_loader, & ts, 0, name, ep_dir, max_sample_slice,  indices,   {indices.size(), 192,192});
      
   
            }

        
        }


    private:
        torch::Tensor merge_coil(torch::Tensor sequence)
        {
            int dim = sequence.sizes().size();
            // [10,32,192,192,2]
            auto output = torch::view_as_complex(sequence);
            torch::Tensor weight;
            if(dim == 5)
            {
                weight = coil_weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
            }
            else if(dim == 6)
            {
                weight = coil_weight.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
            }
            else {
                CHECK(false) << "please check the sequene dim should be 5 or 6" << std::endl;
            }
            output = (output * torch::conj(output.slice(-4,0,1)) * weight ).sum(-3);
            // [10,192,192]
            output = output/(torch::abs(output.slice(-3,0,1)).sqrt());
            return output;

        }
        void SavePNG(torch::Tensor im, std::string image_name)
        {
            CHECK_EQ(im.dim(),2);
            im = im.rot90(3);
            im /= im.max();
            auto im1 = TensorToImage<float> (im.unsqueeze(0).unsqueeze(0));
            auto colorized = ImageTransformation::ColorizeTurbo(im1);
            TemplatedImage<unsigned short> im1_new(im1.dimensions());
            for(int i : im1.rowRange())
            {
                for(int j : im1.colRange())
                {
                    im1_new(i,j) = im1(i,j) * std::numeric_limits<unsigned short>::max();
                }
            }
            im1_new.save(image_name);
        }


    protected:

        // PSNR loss_function_psnr = PSNR(0,1);
        std::shared_ptr<CombinedParams> params;
        std::string experiment_dir;

        std::vector<TrainScene> scenes;

        torch::Tensor coil_weight;
        std::vector<int> temp_train_indices;
        std::vector<int> temp_test_indices;
    public:
        std::shared_ptr<TensorBoardLogger> tblogger;
};





CombinedParams LoadParamsHybrid(int argc, const char** argv)
{
    CLI::App app{"Train on your Scenes", "train"};

    std::string config_file;
    app.add_option("--config", config_file)->required();
    // std::cout << config_file << std::endl;
    // SAIGA_ASSERT(std::filesystem::exists(config_file));
    auto params = CombinedParams();
    params.Load(app);


    try
    {
        app.parse(argc, argv);
    }
    catch (CLI::ParseError& error)
    {
        std::cout << "Parsing failed!" << std::endl;
        std::cout << error.what() << std::endl;
        CHECK(false);
    }

    std::cout << "Loading Config File " << config_file << std::endl;
    params.Load(config_file);
    app.parse(argc, argv);

    return params;
}

int main(int argc, const char* argv[])
{

    int device_num = 1;
    devices.reserve(device_num);


    for (const auto index : c10::irange(device_num)) 
    {
        devices.emplace_back(torch::kCUDA, static_cast<torch::DeviceIndex>(index));
    }
    std::cout << "device size " << devices.size() << devices.front() << std::endl;
    auto params = std::make_shared<CombinedParams>(LoadParamsHybrid(argc, argv));

    std::string experiment_dir;
    if(params->train_params.use_abs_expdir)
    {
        experiment_dir = params->train_params.experiment_dir;
    }
    else
    {
        experiment_dir = PROJECT_DIR.append(params->train_params.exp_dir_name);
    }
    // std::filesystem::create_directories(experiment_dir);
    experiment_dir = experiment_dir + "/" + params->train_params.ExperimentString() + "/";
    if(params->train_params.validate_on)
    {
        if(std::filesystem::exists(params->train_params.checkpoint_directory))
        experiment_dir = params->train_params.checkpoint_directory + "/../";
    }

    std::filesystem::create_directories(experiment_dir);

    // log_ofs = output_stream((experiment_dir+"/log_"+getCurrentDateTime("date")+".txt") );
    log_ofs.file_name = (experiment_dir+"/log_"+getCurrentDateTime("date")+".txt");
    log_ofs.init();

    params->train_params.experiment_dir = experiment_dir;

    if(params->train_params.validate_on)
    {
        if(std::filesystem::exists(experiment_dir + "/params_validate.ini"))
        std::filesystem::remove(experiment_dir + "/params_validate.ini");
        params->Save(experiment_dir + "/params_validate.ini");
    }
    else
    {
        params->Save(experiment_dir + "/params.ini");
    }
    log_ofs << "params geometry type 2 " << params->net_params.geometry_type << std::endl;

    Trainer trainer(params, experiment_dir);
    std::cout << "input batch size " << params->train_params.batch_size << std::endl;

    std::string args_combined;
    for(int i = 0; i < argc; ++i)
    {
        args_combined += std::string(argv[i]) + " ";
    }
    trainer.tblogger->add_text("arguments", 0 , args_combined.c_str());

    int batch_size_scale = std::max((int)devices.size(),2);
    params->train_params.batch_size *= batch_size_scale;
    trainer.ValidateStep();

    log_ofs.close();

    return 0;
}