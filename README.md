# GOAnnotator

Official code for GOAnnotator

# Requirements

python=3.8

sentence-transformers == 2.2.2

pyserini == 0.10.1.0

pygaggle == 0.0.3.1

faiss-cpu == 1.7.4

numpy == 1.23.5

nltk == 3.8.1

# data process

build pubmed index:

```shell
python src/extract_medline_index.py
```

# inference

```shell
cd experiments/scripts
bash run_GOR2023.sh
```