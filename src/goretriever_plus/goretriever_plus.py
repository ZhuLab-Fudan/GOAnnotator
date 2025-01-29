import os
import json
import numpy as np
from tqdm import tqdm
from pyserini.search import SimpleSearcher as LuceneSearcher
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from sentence_transformers import CrossEncoder
from transformers import T5ForConditionalGeneration
import nltk
import torch.nn as nn

from src.utils import TASK_DEFINITIONS, extract_for_train, Config
from utils import get_text


class GORetrieverPlus:
    """Class for GO annotation retrieval and reranking."""

    def __init__(self, config: Config, save_dir, task, data, input, pro_num: int=3, filter_rank: bool=True):
        """
        Initialize the GORetrieverPlus class.
        """
        
        self.save_dir = save_dir
        self.data = data
        self.input = input
        self.filter_rank = filter_rank
        self.pro_num=pro_num

        # Load metadata
        metadata = self._load_metadata(config.METADATA_PATH)
        self.proid2name = dict(map(lambda item: (item[0], extract_for_train(item[1])), metadata.items()))
        self.pmid2text = self._load_metadata(config.PMID2TEXT_FILE)
        self.pro_searcher = LuceneSearcher(config.PRO_INDEX_PATH)
        self.pmid_searcher = LuceneSearcher(config.PMID_INDEX_PATH)
        self.go_searcher = LuceneSearcher(config.GO_INDEX_PATH)

        # Initialize models
        self.t5_reranker = self._initialize_t5_model()
        self.cross_encoder = CrossEncoder(config.MODEL_PATH.format(task), max_length=512)
        self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    @staticmethod
    def _load_metadata(file_path):
        """Load metadata from a .npy file."""
        return np.load(file_path, allow_pickle=True).item()

    @staticmethod
    def _initialize_t5_model():
        """Initialize the T5 reranker model and tokenizer."""
        model = T5ForConditionalGeneration.from_pretrained("castorini/monot5-base-med-msmarco")
        model = nn.DataParallel(model).cuda().module
        tokenizer = MonoT5.get_tokenizer("t5-base")
        return MonoT5(model, tokenizer)

    def data_extract(self):
        """
        Extract relevant sentences from PubMed abstracts for proteins.
        Uses MonoT5 reranker to select the most relevant sentences.

        Returns:
            Dictionary mapping protein IDs to extracted text.
        """
        save_file = os.path.join(self.save_dir, "caches", f"{self.task}_{self.data}_t5_texts.npy")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        
        score_file = save_file.replace("texts", "texts_scores.npy")

        if os.path.exists(save_file):
            print(f"Loading extracted data from: {save_file}")
            return self._load_metadata(save_file)

        pro2text = {}
        text_scores = {}
        pros = set()

        with open(self.input) as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Extracting sentences"):
                pro, pmid = line.split()[:2]
                pmid = pmid.strip()

                # Filter based on ranking threshold
                if self.filter_rank:
                    score = float(line.split()[2].strip())
                    if pro not in pros:
                        threshold = min(0.5, score)
                        pros.add(pro)
                    if score < threshold:
                        continue

                # Get protein name
                if pro not in self.proid2name:
                    print(f"Missing Protein Name: {pro}")
                    continue

                # Get document content
                if pmid in self.pmid2text:
                    text = self.pmid2text[pmid]
                else:
                    try:
                        text = json.loads(self.pmid_searcher.doc(pmid).raw())["contents"]
                    except AttributeError:
                        text = get_text('https://pubmed.ncbi.nlm.nih.gov/'+pmid)
                        self.pmid2text[pmid] = text
                        continue

                sentences = self.tokenizer.tokenize(text)
                if pro not in pro2text:
                    pro2text[pro] = []

                # Skip documents with only one sentence
                if len(sentences) <= 1:
                    print(f"Only one sentence in document: {pmid}")
                    continue

                pro2text[pro].extend(sentences)

        # Apply T5 reranker
        for pro in tqdm(pro2text.keys(), desc="Reranking sentences"):
            query = f"What is the {TASK_DEFINITIONS[self.task]} of protein {self.proid2name[pro]}?"
            sentences = pro2text[pro]

            if len(sentences) < 3:
                print(f"Too little literature data for: {pro}")
                continue

            texts = [Text(sentence, {}, 0) for sentence in sentences]
            scores = self.t5_reranker.rerank(Query(query), texts)

            reranked_scores = {sentences[i]: float(scores[i].score) for i in range(len(scores))}
            text_scores[pro] = reranked_scores

            # Select top sentences
            top_sentences = sorted(reranked_scores, key=reranked_scores.get, reverse=True)[:len(reranked_scores)//2]
            pro2text[pro] = ' '.join(top_sentences)

        np.save(score_file, text_scores)
        np.save(save_file, pro2text)
        print(f"Data extraction completed! Saved to {save_file}")

        return pro2text

    def retrieval(self):
        """
        Retrieve GO annotations for proteins.
        Returns:
            Dictionary mapping protein IDs to retrieved GO terms.
        """
        save_file = os.path.join(self.save_dir, "caches", f"{self.task}_{self.data}_retrieval.npy")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        if os.path.exists(save_file):
            print(f"Loading retrieval data from: {save_file}")
            return self._load_metadata(save_file)

        pro2text = self.data_extract()
        pro2go = self._load_metadata(self.config.TASK_PRO2GO_FILE.format((self.task)))
        retrieval_dict = {}

        for proid in tqdm(pro2text, desc="Retrieving GO terms"):
            try:
                proname = json.loads(self.pro_searcher.doc(proid).raw())["contents"]
            except AttributeError:
                print(f"Missing Protein Name: {proid}")
                continue

            results = self.pro_searcher.search(proname, 3000)
            k = 0
            res = []
            for result in results:
                if k == self.pro_num:
                    break
                doc = json.loads(result.raw)
                if doc["id"] in pro2go:
                    res.extend(pro2go[doc["id"]])
                    k+=1

            retrieval_dict[proid] = res
        print("Write GO Terms Retrieval Results:", save_file)
        np.save(save_file, retrieval_dict)
        return retrieval_dict

def rerank(self):
    """
    Perform reranking on retrieved GO terms using a cross-encoder model.
    Saves the final reranked results in a text file.
    """
    retrieval_data = self.retrieval()
    pro2text = self.data_extract()
    score_dict = os.path.join(self.save_dir, "caches", f"{self.task}_{self.data}_t5_scores.npy")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Load cached scores if available
    if os.path.exists(score_dict):
        print(f"Loading score cache from: {score_dict}")
        score = self._load_metadata(score_dict)
    else:
        score = {}

    # Initialize GO document retriever
    data = []

    print("Starting reranking process...")
    for proid in tqdm(retrieval_data, desc="Reranking GO terms"):
        predicts = []
        goids = []

        # Get protein name
        try:
            proname = self.proid2name[proid]
        except KeyError:
            print(f"Missing protein name: {proid}")
            continue

        # Construct query
        try:
            query = f"The protein is \"{proname}\", the document is \"{pro2text[proid]}\"."
        except KeyError:
            print(f"Text data missing for protein: {proid}")
            continue

        # Retrieve GO term documents and prepare inputs for reranking
        for goid in retrieval_data[proid]:
            if not retrieval_data[proid]:
                print(f"No retrieval GO terms found for: {proid}")
                continue

            if not score.get(proid):
                score[proid] = {}
            elif score[proid].get(goid):
                continue  # Skip if score already cached

            # Fetch GO document content
            try:
                contents = json.loads(self.go_searcher.doc(goid.replace("GO:", "").strip()).raw())["contents"]
            except AttributeError:
                print(f"GO term missing in index: {goid}")
                continue

            goids.append(goid)
            predicts.append([query, contents])

        # Skip if there are no new predictions to score
        if not predicts:
            print(f"Score cache used for protein: {proid}")
            continue

        # Perform reranking using the cross-encoder
        scores = self.cross_encoder.predict(predicts, batch_size=96, show_progress_bar=False)
        for i, goid in enumerate(goids):
            score[proid][goid] = f"{scores[i]:.3f}"

    # Save updated scores to cache
    np.save(score_dict, score)
    print(f"Reranking scores saved to: {score_dict}")

    # Generate final ranked results
    print("Generating reranked results...")
    for proid in tqdm(retrieval_data, desc="Finalizing reranked results"):
        if not retrieval_data[proid]:
            print(f"Retrieval error for protein: {proid}")
            continue
        if proid not in score:
            print(f"Score missing for protein: {proid}")
            continue

        res = {goid: score[proid].get(goid, "0") for goid in retrieval_data[proid]}
        sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=True)[:50]

        for goid, s in sorted_res:
            data.append(f"{proid}\t{goid}\t{s}\n")

    # Save final reranked results
    save_file = os.path.join(self.save_dir, f"{self.task}_t5_{self.data}_rerank.txt")
    if self.args.pro != "0":
        save_file = save_file.replace("_rerank.txt", f"_rerank_pro_{self.pro_num}.txt")

    with open(save_file, "w") as wf:
        wf.writelines(data)

    print(f"Rerank results saved to: {save_file}")
    print("Reranking process completed!")