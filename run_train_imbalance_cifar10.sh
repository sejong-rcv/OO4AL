#!/bin/bash
DIR_PATH_NUM=1
GPU=0
DATE=`date +%m-%d-%T`
INIT_SIZE=100
BUDGET=100
SUBSET=$((BUDGET * 10))
DATA=cifar10im
CLASSES=10
APP=OO4AL # ['Random', 'UncertainGCN', 'CoreSet', 'LL4AL','VAAL','TA-VAAL', 'OO4AL', 'Hierarchical' 'TS'] Please select an option.
for SEED in 0 1 2 3 4
do
    argv=( )
    argv+=( --save_name jobs/${DATA}__approach-${APP}/Init${INIT_SIZE}\_S${SUBSET}_B${BUDGET}/${METHOD}/Seed-${SEED} )
    argv+=( --initial ${INIT_SIZE} )
    argv+=( --selected ${BUDGET} )
    argv+=( --sampling_num ${SUBSET} )
    argv+=( --data_path cifar10 )
    argv+=( --dataset ${DATA} )
    argv+=( --class_num ${CLASSES} )
    argv+=( --seed ${SEED} )
    argv+=( --approach ${APP} )
    argv+=( --milestones 160 )
    argv+=( --lossnet_feature 32 16 8 4 )


    echo OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU} python main.py "${argv[@]}"
    echo ""
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU} python main.py "${argv[@]}"
done