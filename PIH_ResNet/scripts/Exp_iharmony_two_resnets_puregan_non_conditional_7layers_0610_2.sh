#!/bin/bash

# Network hyperparameters
device=2
lr=3e-5
batch_size=16
date=202206102
name=iharmony_puregan_no_conditional_7layerD
workers=8
model_name=exp_${date}_batch_size_$((batch_size))_lr_${lr}_${name}_device_${device}

# Set folder names
dir_data=/mnt/localssd/Image_Harmonization_Dataset/
dir_log=/home/kewang/sensei-fs-symlink/users/kewang/projects/PIH/PIH_ResNet/results/$model_name



CUDA_VISIBLE_DEVICES=$device python PIH_train_GAN.py --datadir $dir_data \
                       -g 0 \
                       --logdir $dir_log \
                       --bs $batch_size \
                       --lr $lr \
                       --lrd 1e-5 \
                       --force_train_from_scratch \
                       --gan-weight 1 \
                       --noreconloss \
                       --tempdir \
                       temp_results_gan_no_conditional_7layer_iharmony_largelr \
                       --ihd \
                       --workers $workers

