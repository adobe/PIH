#!/bin/bash

# Network hyperparameters
device=4
lr=4e-5
lrd=4e-5
batch_size=8
date=20221016_a100_4_raw_lowdim_L1_reconwithgan_brush_offset0_upsample_lessskip_agg_Ihd_composite_all_scheduler_eff
reconweight=None
training_ratio=1
lutdim=16
inputdimD=3
recon_ratio=1
recon_weight=1 ## Used here

name=iharmony_${inputdimD}_ratio_${training_ratio}_${recon_ratio}_reconweight_${recon_weight}

model_name=exp_${date}_batch_size_$((batch_size))_lr_${lr}_${name}_device_${device}

# Set folder names
dir_data=/mnt/localssd/Ihd_composite_all/train/
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
                       --pl-dim 64 \
                       --pairaugment \
                       --purepairaugment \
                       --lowdim \
                       --ganlossmask \
                       --reconwithgan \
                       --reconweight ${recon_weight} \
                       --masking \
                       --brush \
                       --maskoffset 0 \
                       --swap \
                       --onlyupsample \
                       --joint \
                       --lessskip \
                       --aggupsample \
                       --iharmdata \
                       --scheduler \
                       --returnraw \
                       --twoinputs \
                       --effbool \
                    #    --lowres \
                       




