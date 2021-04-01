#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
if [ ! -d $1 ]
then 
    echo "error: DATASET_PATH=$1 is not a directory"
    exit 1
fi 

if [ ! -f $2 ]
then 
    echo "error: CHECKPOINT_PATH=$2 is not a file"
    exit 1
fi 

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

echo "start evaluation for device $DEVICE_ID"
python eval.py --dataset_path=$1 --checkpoint_path=$2 &> eval.log &