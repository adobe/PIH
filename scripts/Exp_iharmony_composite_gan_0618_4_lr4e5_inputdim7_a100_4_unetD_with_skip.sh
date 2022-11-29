#!/bin/bash

# Network hyperparameters
device=6
lr=4e-5
lrd=4e-5
batch_size=12
date=20220618_a100_4_unetD_withmask_with_skip
inputdim=7
reconweight=Nonee
training_ratio=1.0
name=iharmony_compositegan_dim_${inputdim}_reconloss_${reconweight}_ration_${training_ratio}_nomask_noskip

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
                       --unetd 7\
                       --unetmask \


