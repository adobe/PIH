#!/bin/bash

# Network hyperparameters
device=0
lr=4e-5
lrd=4e-5
batch_size=8
date=20221103_a100_0_raw_lowdim_L1_reconwithgan_brush_offset0_upsample_lessskip_agg_Ihd_composite_all_scheduler_resnet50_pixel
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



CUDA_VISIBLE_DEVICES=$device python PIH_train_compositeGAN_l1.py --datadir $dir_data \
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
                       --swap \
                       --joint \
                       --aggupsample \
                       --iharmdata \
                       --scheduler \
                       --returnraw \
                       --twoinputs \
                       --maskingcp /home/kewang/sensei-fs-symlink/users/kewang/projects/PIH/PIH_ResNet/results/exp_20221016_a100_3_raw_lowdim_L1_reconwithgan_094_brush_offset075_upsample_lessskip_agg_Ihd_composite_all_scheduler_lowres_batch_size_8_lr_4e-5_iharmony_3_ratio_1_1_reconweight_1_device_3/checkpoints/ckpt_g49.pth \
                     #   --effbool \
                     
                     #   --colorjitter \

                    #    --lowres \
                       




