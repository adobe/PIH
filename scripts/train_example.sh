# Copyright 2023 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

# Network hyperparameters
device=0
lr=4e-5
lrd=4e-5
batch_size=8
date=Demo_train_example
reconweight=None
training_ratio=1
inputdimD=3
recon_ratio=0.5
recon_weight=0.92 ## Used here

name=iharmony_${inputdimD}_ratio_${training_ratio}_${recon_ratio}_reconweight_${recon_weight}

model_name=exp_${date}_batch_size_$((batch_size))_lr_${lr}_${name}_device_${device}

# Set folder names
dir_data=/mnt/localssd/LR_data/train/


CUDA_VISIBLE_DEVICES=$device python PIH_train.py --datadir $dir_data \
                       -g 0 \
                       --onlysaveg \
                       --bs $batch_size \
                       --lr $lr \
                       --lrd $lrd \
                       --force_train_from_scratch \
                       --tempdir \
                       $model_name \
                       --workers 8 \
                       --trainingratio ${training_ratio} \
                       --unetd \
                       --inputdimD ${inputdimD} \
                       --nocurve \
                       --reconratio ${recon_ratio} \
                       --piecewiselinear \
                       --pl-dim 32 \
                       --pairaugment \
                       --purepairaugment \
                       --lowdim \
                       --ganlossmask \
                       --reconwithgan \
                       --reconweight ${recon_weight} \
                       --masking \
                       --brush \
                       --maskoffset 0.5 \
                       --swap \
                       --onlyupsample \
                       --joint \
                       --lessskip \
                       --aggupsample \
                       --scheduler \
                       --colorjitter \

                       




