#!/bin/bash

echo Checkpoint: $1
echo Model_Name: $2
echo GPU: $3

CUDA_VISIBLE_DEVICES=$3

mkdir /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/
# /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/File_for_testing_composite_realhm/
 
CUDA_VISIBLE_DEVICES=$3 python PIH_test_compositeGAN_masking.py --datadir /mnt/localssd/File_for_testing_composite_realhm/ \
                                           -g 0 \
                                           --checkpoints $1 \
                                           --tmp_results /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/results_testing/ \
                                           --bs 1 \
                                           --composite \
                                           --num-testing 500 \
                                           --nocurve \
                                           --piecewiselinear \
                                           --masking \
                                           --brush \
                                           --maskoffset 0.6 \
                                           --swap \
                                                                                    #   --pihnetbool \
                                        #    --lut \

                                                                                   #    --lut \
                                        #    --onlyupsample \
                                        #    --maskconvkernel 3 \

#                                           #  --onlyupsample \
#                                           #  --maskconvkernel 3 \
#                                           #  --maskoffset 0 \

#                                           #  --onlyupsample \
#                                           #  --maskconvkernel 3 \
#                                           #  --maskoffset 0 \
#                                           #  --mask

#                                         #    --onlyupsample \
#                                         #    --nosig \


mkdir /home/kewang/website_michael/results/$2/

cd /home/kewang/website_michael/ttools2-main/ttools2/scripts/

python /home/kewang/website_michael/ttools2-main/ttools2/scripts/image_gallery.py /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/results_testing/ \
                                                                                  /home/kewang/website_michael/results/$2/results_testing/ \
                                                                                  --order bg mask original intermediate results real curves\
                                                                                  --name $2-results_testing


