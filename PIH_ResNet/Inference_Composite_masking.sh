#!/bin/bash

echo Checkpoint: $1
echo Model_Name: $2
echo GPU: $3
echo mask offset: $4

echo $5

if [ $5 == realhm ]
then
   a=File_for_testing_composite_realhm
elif [ $5 == realhr ]
then
   a=File_for_testing_composite_realhr
elif [ $5 == iharm ]
then
   a=File_for_testing_composite_iharm
elif [ $5 == LR ]
then
   a=File_for_testing_composite_LR_2000

elif [ $5 == LRself ]
then
   a=File_for_testing_composite_LR_self_2000


elif [ $5 == self ]
then
   a=File_for_testing_self

elif [ $5 == realself ]
then
   a=File_for_testing_composite_realself

elif [ $5 == adobe ]
then
   a=File_for_testing_composite_adobereal

else
   a=0
fi

echo Data stream: $a

CUDA_VISIBLE_DEVICES=$3

mkdir /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/
# /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/File_for_testing_composite_realhm/
#  /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/File_for_testing_composite_LR_self_2000/

CUDA_VISIBLE_DEVICES=$3 python PIH_test_compositeGAN_masking.py --datadir /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/$a \
                                           -g 0 \
                                           --checkpoints $1 \
                                           --tmp_results /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/results_testing/ \
                                           --bs 1 \
                                           --composite \
                                           --num-testing 50000 \
                                           --nocurve \
                                           --piecewiselinear \
                                           --masking \
                                           --brush \
                                           --maskoffset $4 \
                                           --swap \
                                           --onlyupsample \
                                           --twoinputs \
                                           --aggupsample \
                                           --dim 64 \
                                          # --effbool \

                                          # --lowres \
                                          # --effbool \
                                          # --dim 64 \
                                          # --lut \
                                          # --lut-dim 16 \


                                        #  --vitbool \
                                           
                                        #    --pihnetbool \
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


