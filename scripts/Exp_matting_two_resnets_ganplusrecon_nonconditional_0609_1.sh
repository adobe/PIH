#!/bin/bash

# Network hyperparameters
device=3
lr=1e-5
batch_size=16
date=202206093
name=matting_ganplusrecon_1_non_conditioanl_7layer

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
                       --gan-weight 0.01 \
                        --tempdir \
                       temp_results_ganplusrecon001_non_conditional_7layer \
