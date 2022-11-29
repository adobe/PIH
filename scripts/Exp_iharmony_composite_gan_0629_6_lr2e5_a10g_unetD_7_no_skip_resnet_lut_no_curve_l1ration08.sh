#!/bin/bash

# Network hyperparameters
device=3
lr=2e-5
lrd=2e-5
batch_size=8
date=20220629_a10g_0_unetD_7_no_skip_resnet_lut_dim_16_maskinput_no_curve_l1ratio02
reconweight=None
training_ratio=1
lutdim=16
inputdimD=7
recon_ratio=0.8

name=iharmony_compositegan_D_${inputdimD}_ratio_${training_ratio}_noskip_lutdim${lutdim}_no_curve_reconratio_${recon_ratio}

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


