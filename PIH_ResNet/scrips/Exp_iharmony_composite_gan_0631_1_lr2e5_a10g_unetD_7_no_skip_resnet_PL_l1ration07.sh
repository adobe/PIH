#!/bin/bash

# Network hyperparameters
device=2
lr=2e-5
lrd=2e-5
batch_size=8
date=20220629_a10g_0_unetD_7_no_skip_resnet_maskinput_pl32_l1ratio02
reconweight=None
training_ratio=1
lutdim=16
inputdimD=7
recon_ratio=0.7

name=iharmony_compositegan_D_${inputdimD}_ratio_${training_ratio}_noskip_PL32_reconratio_${recon_ratio}

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
                       --workers 8 \
                       --trainingratio ${training_ratio} \
                       --unetd \
                       --unetdnoskip \
                       --inputdimD ${inputdimD} \
                       --lut \
                       --lut-dim ${lutdim} \
                       --nocurve \
                       --reconratio ${recon_ratio} \
                       --piecewiselinear \


