#!/bin/bash

# Network hyperparameters
device=1
lr=1e-4
lrd=1e-4
batch_size=12
date=20220617_a100_1_unetD_withmask_no_skip
inputdim=3
reconweight=None
training_ratio=0.2
name=iharmony_compositegan_7layerD_unet_test_with_mask_no_mask_input_${inputdim}_reconloss_${reconweight}_ration_${training_ratio}_unet_D_nomask_noskip

model_name=exp_${date}_batch_size_$((batch_size))_lr_${lr}_${name}_device_${device}

# Set folder names
dir_data=/mnt/localssd/Ihd_real_composite/train/
dir_log=/home/kewang/sensei-fs-symlink/users/kewang/projects/PIH/PIH_ResNet/results/$model_name



CUDA_VISIBLE_DEVICES=$device python PIH_train_compositeGAN.py --datadir $dir_data \
                       -g 0 \
                       --logdir $dir_log \
                       --bs $batch_size \
                       --lr $lr \
                       --lrd $lrd \
                       --force_train_from_scratch \
                       --tempdir \
                       $model_name \
                       --workers 12 \
                       --unet \
                       --inputdim $inputdim \
                       --trainingratio ${training_ratio} \
                       --unetd \
                       --unetdnoskip \
                       --unetmask \


