conda create -y -n diffunfold python=3.9

CONDA_PATH=~/anaconda3/

CONDA_PATH=~/miniconda/
# export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"   # a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1
# https://developer.nvidia.com/cuda-gpus  8.9 need cuda>=11.8
export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0;8.6;8.9" # 4090: 8.9, 3090: 8.6, rtx a6000:8.6 a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1
conda activate diffunfold
conda install -y -c conda-forge libstdcxx-ng=9 coin-or-cbc libgomp=9 glog gflags protobuf=3.19
conda install -c anaconda make
export TMPDIR=/project/cigserver4/export1/yuanhao/env/tmp
conda install -y cuda  -c nvidia/label/cuda-12.4
conda install nvidia/label/cuda-12.4.0::cuda-toolkit

conda install -y -c conda-forge cudnn=8.9.2
conda install -y magma-cuda110 -c pytorch
conda install -y astunparse numpy=1.22.3  ninja pyyaml mkl mkl-include cmake=3.22.1 cffi typing_extensions future six requests dataclasses setuptools tensorboard configargparse
conda install -y boost xorg-libxrandr xorg-libxinerama xorg-libxcursor xorg-libxi ncurses glfw glew freetype nibabel imageio re2 freeimage=3.18 libffi=3.3 -c conda-forge
pip install scipy==1.12
conda install -y freeimageplus -c dlr-sc
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip -O  libtorch.zip
unzip libtorch.zip -d .
export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

cp -rv libtorch/ $CONDA/lib/python3.9/site-packages/torch/

cd saiga_patch
sh saiga_patch.sh
cd ..
export TMPDIR=/tmp
export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export GCC_ROOT=/home/yuanhao/Downloads/software/gcc-9
export CUDA_HOME=$CONDA
# export CUDA_HOME=$CONDA/pkgs/cuda-toolkit
export CUDA_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export CC=$GCC_ROOT/bin/gcc
export CXX=$GCC_ROOT/bin/g++
export CUDAHOSTCXX=$GCC_ROOT/bin/g++
# cmake -DCMAKE_PREFIX_PATH="${CONDA};${CONDA}/lib/python3.9/site-packages/torch/;" ..
cmake -DCMAKE_PREFIX_PATH="${CONDA};./External/libtorch/" ..
# cmake -DCMAKE_PREFIX_PATH="${CONDA};./libtorch/" ..

make -j64
