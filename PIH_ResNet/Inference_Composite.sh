#!/bin/bash

echo Checkpoint: $1
echo Model_Name: $2
echo GPU: $3

CUDA_VISIBLE_DEVICES=$3

mkdir /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/

CUDA_VISIBLE_DEVICES=$3 python PIH_test_compositeGAN.py --datadir /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/File_for_testing_composite_LR_2000/ \
                                           -g 0 \
                                           --checkpoints $1 \
                                           --tmp_results /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/results_testing/ \
                                           --bs 1 \
                                           --composite \
                                           --lut \
                                           --lut-dim 16 \
                                           --num-testing 2000 \
                                           --nocurve \
                                           --piecewiselinear \

mkdir /home/kewang/website_michael/results/$2/

cd /home/kewang/website_michael/ttools2-main/ttools2/scripts/

python /home/kewang/website_michael/ttools2-main/ttools2/scripts/image_gallery.py /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/results_testing/ \
                                                                                  /home/kewang/website_michael/results/$2/results_testing/ \
                                                                                  --order bg mask original results real curves\
                                                                                  --name $2-results_testing


