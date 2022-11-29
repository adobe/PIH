#!/bin/bash

# Network hyperparameters
device=2
lr=1e-5
batch_size=12
date=202206151
name=iharmony_compositegan_7layerD_unet_test_sgd

model_name=exp_${date}_batch_size_$((batch_size))_lr_${lr}_${name}_device_${device}

# Set folder names
dir_data=/mnt/localssd/Ihd_real_composite/train/
dir_log=/home/kewang/sensei-fs-symlink/users/kewang/projects/PIH/PIH_ResNet/results/$model_name



CUDA_VISIBLE_DEVICES=$device python PIH_train_compositeGAN.py --datadir $dir_data \
                       -g 0 \
                       --logdir $dir_log \
                       --bs $batch_size \
                       --lr $lr \
                       --force_train_from_scratch \
                       --tempdir \
                       $model_name \
                       --workers 12 \
                       --unet \
                       --sgd \
