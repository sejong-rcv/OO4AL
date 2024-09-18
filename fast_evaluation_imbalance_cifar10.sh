#!/bin/bash
GPU=0            # enter your gpu number to use
DATA=cifar10im
CLASSES=10
CHECKPOINT=jobs/cifar10lt/Init100_S1000_B100/Seed-{}/active_resnet18_cifar10_cycle{}.pth
argv=( )
argv+=( --data_path cifar10 )
argv+=( --dataset ${DATA} )
argv+=( --class_num ${CLASSES} )
argv+=( --checkpoint ${CHECKPOINT} )
argv+=( --approach OO4AL )
argv+=( --batch_size 128 )
argv+=( --workers 0 )

echo OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU} python eval.py "${argv[@]}"
echo ""
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU} python eval.py "${argv[@]}"