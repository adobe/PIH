#!/bin/bash

# Network hyperparameters
device=4
lr=2e-5
lrd=2e-5
batch_size=8
date=20220826_a100_4_unetD_3_with_skip_resnet_maskinput_pl64_gan_loss_mask_lowdim_L105_reconwithgan_05_effnet_32
reconweight=None
training_ratio=1
lutdim=16
inputdimD=3
recon_ratio=0.5
recon_weight=0.75 ## Used here

name=iharmony_compositegan_D_${inputdimD}_ratio_${training_ratio}_noskip_PL32_reconratio_${recon_ratio}_reconweight_${recon_weight}

model_name=exp_${date}_batch_size_$((batch_size))_lr_${lr}_${name}_device_${device}

# Set folder names
dir_data=/mnt/localssd/LR_data/train/
dir_log=/home/kewang/sensei-fs-symlink/users/kewang/projects/PIH/PIH_ResNet/results/$model_name



CUDA_VISIBLE_DEVICES=$device python PIH_train_compositeGAN.py --datadir $dir_data \
                       -g 0 \
                       --onlysaveg \
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
                       --effbool \

                    #    --masking \
                    #    --brush \
                    #    --maskoffset 0.7 \
                    #    --swap \
                    #    --joint \
                    #    --effbool \

                       



