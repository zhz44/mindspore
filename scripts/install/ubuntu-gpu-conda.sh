#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Prepare and Install mindspore gpu by conda on Ubuntu 18.04.
#
# This file will:
#   - change deb source to huaweicloud mirror
#   - install mindspore dependencies via apt like gcc, libgmp
#   - install conda and set up environment for mindspore
#   - install mindspore-gpu by conda
#   - compile and install Open MPI if OPENMPI is set to on.
#
# Augments:
#   - PYTHON_VERSION: python version to install. [3.7(default), 3.9]
#   - MINDSPORE_VERSION: mindspore version to install, default 1.6.0
#   - CUDA_VERSION: CUDA version to install. [10.1(default), 11.1]
#   - OPENMPI: whether to install optional package Open MPI for distributed training. [on, off(default)]
#
# Usage:
#   Run script like `bash -i ./ubuntu-gpu-conda.sh`.
#   To set augments, run it as `PYTHON_VERSION=3.9 CUDA_VERSION=11.1 OPENMPI=on bash -i ./ubuntu-gpu-conda.sh`.

set -e

PYTHON_VERSION=${PYTHON_VERSION:-3.7}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.6.0}
CUDA_VERSION=${CUDA_VERSION:-10.1}
OPENMPI=${OPENMPI:-off}

available_py_version=(3.7 3.9)
if [[ " ${available_py_version[*]} " != *" $PYTHON_VERSION "* ]]; then
    echo "PYTHON_VERSION is '$PYTHON_VERSION', but available versions are [${available_py_version[*]}]."
    exit 1
fi
available_cuda_version=(10.1 11.1)
if [[ " ${available_cuda_version[*]} " != *" $CUDA_VERSION "* ]]; then
    echo "CUDA_VERSION is '$CUDA_VERSION', but available versions are [${available_cuda_version[*]}]."
    exit 1
fi

# add value to environment variable if value is not in it
add_env() {
    local name=$1
    if [[ ":${!name}:" != *":$2:"* ]]; then
        echo -e "export $1=$2:\$$1" >> ~/.bashrc
    fi
}

install_conda() {
    conda_file_name="Miniconda3-py3${PYTHON_VERSION##*.}_4.10.3-Linux-$(arch).sh"
    cd /tmp
    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/$conda_file_name
    bash $conda_file_name -b
    cd -
    . ~/miniconda3/etc/profile.d/conda.sh
    conda init bash
    # setting up conda mirror with tsinghua source
    cat >~/.condarc <<END
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
END
}

# use huaweicloud mirror in China
sudo sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo apt-get update

sudo apt-get install curl gcc-7 libgmp-dev -y

# optional openmpi for distributed training
if [[ X"$OPENMPI" == "Xon" ]]; then
    cd /tmp
    curl -O https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz
    tar xzf openmpi-4.0.3.tar.gz
    cd openmpi-4.0.3
    ./configure --prefix=/usr/local/openmpi-4.0.3
    make
    sudo make install
    set +e && source ~/.bashrc
    set -e
    add_env PATH /usr/local/openmpi-4.0.3/bin
    add_env LD_LIBRARY_PATH /usr/local/openmpi-4.0.3/lib
fi

# install conda
set +e && type conda &>/dev/null
if [[ $? -eq 0 ]]; then
    echo "conda has been installed, skip."
    source "$(conda info --base)"/etc/profile.d/conda.sh
else
    install_conda
fi
set -e

# set up conda env and install mindspore-cpu
env_name=mindspore_py3${PYTHON_VERSION##*.}
declare -A cudnn_version_map=()
cudnn_version_map["10.1"]="7.6.5"
cudnn_version_map["11.1"]="8.1.0"
conda create -n $env_name python=${PYTHON_VERSION} -y
conda activate $env_name
conda install mindspore-gpu=${MINDSPORE_VERSION} \
    cudatoolkit=${CUDA_VERSION} cudnn=${cudnn_version_map[$CUDA_VERSION]} -c mindspore -c conda-forge -y

# check mindspore installation
python -c "import mindspore;mindspore.run_check()"

# check if it can be run with GPU
cd /tmp
cat > example.py <<END
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="GPU")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
END
python example.py
cd -
