#!/bin/bash
export PYTHONPATH=$PYTHONPATH:../../src

DATA="GOR2023_Pubtator"
MAP="../../data/GOR2023/map/pubtator.map"
SAVE_DIR="../results/GOR2023/"

# DATA="GOR2023_GORetriever"
# MAP="../../data/GOR2023/map/swissprot.map"
# SAVE_DIR="../results/GOR2023/"

# DATA="SP2024_Pubtator"
# MAP="../../data/SP2024/map/pubtator.map"
# SAVE_DIR="../results/SP2024/"

# DATA="SP2024_GORetriever"
# MAP="../../data/SP2024/map/swissprot.map"
# SAVE_DIR="../results/SP2024/"

tasks=("mf" "bp" "cc")
pros=(3 3 2)
gpus=(1 2 3)
for i in ${!tasks[@]}; do
    task=${tasks[$i]}
    pro=${pros[$i]}
    gpu=${gpus[$i]}

    log_file="../logs/${task}_gor.log"
    CUDA_VISIBLE_DEVICES=$gpu python ../../src/run_goretrieverplus.py --save_dir $SAVE_DIR --task $task --data $DATA --input $MAP --pro_num $pro --filter_rank True >> $log_file 2>&1 &
done