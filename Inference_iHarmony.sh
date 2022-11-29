#!/bin/bash

echo Checkpoint: $1
echo Model_Name: $2
echo GPU: $3

CUDA_VISIBLE_DEVICES=$3

mkdir /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/


                                           
CUDA_VISIBLE_DEVICES=$3 python PIH_test.py --datadir /mnt/localssd/Image_Harmonization_Dataset/ \
                                           -g 0 \
                                           --checkpoints $1 \
                                           --tmp_results /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/results_testing_iHarmony/ \
                                           --bs 1 \
                                           --num-testing 1000 \
                                           --ihd \


mkdir /home/kewang/website_michael/results/$2/

cd /home/kewang/website_michael/ttools2-main/ttools2/scripts/

python /home/kewang/website_michael/ttools2-main/ttools2/scripts/image_gallery.py /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/results_testing_iHarmony/ \
                                                                                  /home/kewang/website_michael/results/$2/results_testing_iHarmony/ \
                                                                                  --order original mask intermediate results gt\
                                                                                  --name $2-results-testing-iHarmony

# # Network hyperparameters
# device=1
# lr=1e-5
# batch_size=16
# date=202206070
# name=iharmonysimplemodel

# model_name=exp_${date}_batch_size_$((batch_size))_lr_${lr}_${name}

# # Set folder names
# dir_data=/mnt/localssd/Image_Harmonization_Dataset/
# dir_log=/home/kewang/sensei-fs-symlink/users/kewang/projects/PIH/PIH_ResNet/results/$model_name



# CUDA_VISIBLE_DEVICES=$device python PIH_train.py --datadir $dir_data \
#                        -g 0 \
#                        --logdir $dir_log \
#                        --bs $batch_size \
#                        --lr $lr \
#                        --force_train_from_scratch \
#                        --ihd \ 
