#!/bin/bash

# Network hyperparameters
device=3
lr=1e-5
lrd=2e-6
batch_size=12
date=202206142
name=iharmony_compositegan_6layerD

model_name=exp_${date}_batch_size_$((batch_size))_lr_${lr}_lrd_${lrd}_${name}_device_${device}

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
