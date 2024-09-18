#!/bin/bash
DIR_PATH_NUM=1
GPU=0
DATE=`date +%m-%d-%T`
INIT_SIZE=128120  # 10%
BUDGET=64060      # 5%
SUBSET=$((BUDGET * 10))
DATA=imagenet 
CLASSES=1000
APP=OO4AL        # ['Random', 'LL4AL', 'OO4AL'] Please select an option.
Layer=1

for SEED in 0 1 2
do
    argv=( )
    argv+=( --save_name jobs/${DATA}__approach-${APP}/Init${INIT_SIZE}\_S${SUBSET}_B${BUDGET}/${METHOD}/Seed-${SEED} )
    argv+=( --initial ${INIT_SIZE} )
    argv+=( --selected ${BUDGET} )
    argv+=( --sampling_num ${SUBSET} )
    argv+=( --data_path ImageNet2012/ )
    argv+=( --dataset ${DATA} )
    argv+=( --class_num ${CLASSES} )
    argv+=( --seed ${SEED} )
    argv+=( --approach ${APP} )
    argv+=( --layer ${Layer} )
    argv+=( --epoch_num 100 )
    argv+=( --cycle_num 5 )
    argv+=( --workers 8 )
    argv+=( --batch_size 256 )
    argv+=( --epochl 60 )
    argv+=( --wdecay 0.0001 )
    argv+=( --milestones 30 60 90 )
    argv+=( --lossnet_feature 56 28 14 7 )


    echo OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU} python main.py "${argv[@]}"
    echo ""
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU} python main.py "${argv[@]}"
done