#!/bin/bash

# Network hyperparameters
device=1
lr=1e-5
batch_size=16
date=202206092
name=matting_puregan_conditional_7layerD

model_name=exp_${date}_batch_size_$((batch_size))_lr_${lr}_${name}_device_${device}

# Set folder names
dir_data=/mnt/localssd/File_for_training_3/
dir_log=/home/kewang/sensei-fs-symlink/users/kewang/projects/PIH/PIH_ResNet/results/$model_name



CUDA_VISIBLE_DEVICES=$device python PIH_train_GAN.py --datadir $dir_data \
                       -g 0 \
                       --logdir $dir_log \
                       --bs $batch_size \
                       --lr $lr \
                       --force_train_from_scratch \
                       --random_aug \
                       --gan-weight 1 \
                       --noreconloss \
                       --tempdir \
                       temp_results_gan_conditional_7layer \
                       --conditional \
