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

# Data Preprocessing

build pubmed index:

```shell
cd experiments/scripts
bash build_index.sh
```

unzip protein_metadata:

```shell
cd src/dependencies
tar -xzvf protein_metadata.tar.gz protein_metadata.npy
```

# Prediction

For GOR2023/SP2024/TR2024:

```shell
cd experiments/scripts
bash run.sh
```

For predictions using the given map(protein id - pubmed id):

```shell
cd experiments/scripts
bash run_map.sh
```

For GOR2023_PT:

```shell
cd experiments/scripts
bash run_GOR2023_PT.sh
```

For TR2024_SP:

```shell
cd experiments/scripts
bash run_TR2024_SP.sh
```

# Models

| Models     | Huggingface Link     |
|----------|---------------|
| PubRetriever_reranker | [Link](https://huggingface.co/whitneyyan0122/pubretriever-reranker)  |
| GORetriever+_MFO | [Link](https://huggingface.co/whitneyyan0122/GORetriever_plus_mf_PubMedBERT) |
| GORetriever+_BPO | [Link](https://huggingface.co/whitneyyan0122/GORetriever_plus_bp_PubMedBERT) |
| GORetriever+_CCO | [Link](https://huggingface.co/whitneyyan0122/GORetriever_plus_cc_PubMedBERT) |
