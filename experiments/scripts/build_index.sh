# !/bin/bash

log_folder="../logs"
mkdir -p "$log_folder"

start=1
end=100
max_jobs=5
step=50

while [ "$start" -le "$end" ]; do
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 1
    done

    current_end=$((start+step-1))
    
    if [ "$current_end" -gt "$end" ]; then
        current_end="$end"
    fi

    log_file="$log_folder/get_xml-${start}-${current_end}.log"
    
    nohup python ../../src/extract_medline_index.py --start "$start" --end "$current_end" > "$log_file" 2>&1 &
    
    echo "Started process for range $start-$current_end. Log file: $log_file"
    
    start=$((start+step))
done

wait

echo "All processes completed."

nohup python -m pyserini.index.lucene -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 8 \
    -input ../../src/dependencies/pubmed/json \
    -index ../../src/dependencies/2024-pubmed-uniprot-index \
    -storePositions -storeDocvectors -storeRaw > "${log_folder}/index.log" 2>&1 &