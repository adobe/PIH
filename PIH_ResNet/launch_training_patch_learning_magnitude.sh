#!/bin/bash


# Set folder names
dir_data=/home/kewang/target_patch_500_40/patch_data/
device=0
dimension=256
lr=1e-5
temperature=1
batch=256
epochs=200
date=20211014
model_name=train_alma_UFLoss_feature_$((dimension))_features_date_$((date))_temperature_${temperature}_lr${lr}
dir_save=/mikQNAP/NYU_knee_data/knee_train_h5/checkpoints_ufloss_mapping/$model_name


python patch_learning.py --datadir $dir_data\
                 -g $device \
                 --logdir $dir_save \
                 -f $dimension \
                 --learning-rate $lr \
                 --temperature $temperature \
                 --batchsize $batch \
                 -e $epochs \
		 --use_magnitude \
                 --use_mag_augmentation \
