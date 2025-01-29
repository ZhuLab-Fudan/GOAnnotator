# #!/bin/bash

tasks=("mf" "bp" "cc")
pros=(3 3 2)
gpus=(1 2 3)
for i in ${!tasks[@]}; do
    task=${tasks[$i]}
    pro=${pros[$i]}
    gpu=${gpus[$i]}
    MAP="./results/GOR2023/map/${task}.map"
    INPUT="../../data/GOR2023/pid/${task}.pid"
    log_file="logs/${task}.log"

    CUDA_VISIBLE_DEVICES=$gpu python run_pubretriever.py --input $input --output $output --task $task > $log_file 2>&1 

    SAVE_DIR="results/GOR2023/"
    DATA="GOR2023"
    CUDA_VISIBLE_DEVICES=$gpu python run_goretrieverplus.py --save_dir $SAVE_DIR --task $task --data $DATA --input $MAP --pro_num $pro --filter_rank True >> $log_file 2>&1 
done