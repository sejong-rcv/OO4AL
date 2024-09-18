#!/bin/bash
GPU=0            # enter your gpu number to use
DATA=cifar100
CLASSES=100
CHECKPOINT=jobs/cifar100/Init1000_S10000_B1000/Seed-{}/active_resnet18_cifar100_cycle{}.pth
argv=( )
argv+=( --data_path cifar100 )
argv+=( --dataset ${DATA} )
argv+=( --class_num ${CLASSES} )
argv+=( --checkpoint ${CHECKPOINT} )
argv+=( --approach OO4AL )
argv+=( --batch_size 128 )
argv+=( --workers 0 )

echo OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU} python eval.py "${argv[@]}"
echo ""
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU} python eval.py "${argv[@]}"