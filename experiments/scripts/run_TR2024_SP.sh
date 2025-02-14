#!/bin/bash
export PYTHONPATH=$PYTHONPATH:../../src

DATA="TR2024_SP"
mkdir -p ../results/TR2024/map/
tasks=("mf" "bp" "cc")
pros=(3 3 2)
gpus=(1 2 3)
for i in ${!tasks[@]}; do
    task=${tasks[$i]}
    pro=${pros[$i]}
    gpu=${gpus[$i]}
    MAP="../results/TR2024/map/${task}_SP.map"
    INPUT="../../data/TR2024/pid/${task}_SP.pid"
    log_file="../logs/${task}_pub.log"
    (
    CUDA_VISIBLE_DEVICES=$gpu python ../../src/run_pubretriever.py --input $INPUT --output $MAP --task $task > $log_file 2>&1 

    SAVE_DIR="../results/TR2024/"
    log_file="../logs/${task}_gor.log"
    CUDA_VISIBLE_DEVICES=$gpu python ../../src/run_goretrieverplus.py --save_dir $SAVE_DIR --task $task --data $DATA --input $MAP --pro_num $pro --filter_rank True >> $log_file 2>&1 
    ) &
done