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


elif [ $5 == adobe ]
then
   a=File_for_testing_composite_adobereal

elif [ $5 == realself ]
then
   a=File_for_testing_composite_realself


else
   a=0
fi

echo Data stream: $a

CUDA_VISIBLE_DEVICES=$3

mkdir /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/results_images/$2/
# /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/File_for_testing_composite_realhm/
#  /home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/File_for_testing_composite_LR_self_2000/

CUDA_VISIBLE_DEVICES=$3 python PIH_test_compositeGAN_masking_tiny.py --datadir /mnt/localssd/$a \
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
                                           --maskoffset $4 \
                                           --swap \
                                           --onlyupsample \
                                           --aggupsample \
                                           --dim 32 \
                                          #  --lut \
                                          #  --lut-dim 16 \
                                          #  --lowres \
                                          #  --vitbool \
                                          #  --ibn \
                                          #  --effbool \
                                          #  --ibn \
                                          #  --vitbool \
                                          #  --ibn \
#                                           --bgshadow \
                                          # --twoinputs \


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




