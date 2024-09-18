#!/bin/bash
GPU=0            # enter your gpu number to use
DATA=imagenet
CLASSES=1000
CHECKPOINT=jobs/imagenet/Init128120_S640600_B64060/Seed-{}/active_resnet18_imagenet_cycle{}.pth
argv=( )
argv+=( --data_path ImageNet2012 )
argv+=( --dataset ${DATA} )
argv+=( --class_num ${CLASSES} )
argv+=( --checkpoint ${CHECKPOINT} )
argv+=( --approach OO4AL )
argv+=( --batch_size 128 )
argv+=( --workers 8 )
argv+=( --num_of_cycle 5 )


echo OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU} python eval.py "${argv[@]}"
echo ""
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU} python eval.py "${argv[@]}"