import os
from typing import List, Dict
import json
import numpy as np
from tqdm import tqdm
from pyserini.search import SimpleSearcher as LuceneSearcher
from sentence_transformers import CrossEncoder
from transformers import T5ForConditionalGeneration
from pygaggle.rerank.transformer import MonoT5
import nltk
import torch.nn as nn
from src.utils import TASK_DEFINITIONS, extract_for_FIR, extract_for_train
from utils import  get_text 


class GORetrieverPlus:
    """
    A class for GO annotation retrieval and reranking.
    """

    def __init__(
        self,
        model_path: str,
        pro_index_path: str,
        go_index_path: str,
        metadata_path: str,
        pmid2text_path: str,
        task_pro2go_path: str,
        input_file: str,
        task: str = "bp",
        pro: int = 3,
        filter_rank: bool = False,
        save_dir: str = "./results",
        device: str = "cuda",
    ):
        """
        Initialize the GORetrieverPlus class.

        Args:
            model_path (str): Path to the pre-trained CrossEncoder model.
            pro_index_path (str): Path to the Lucene index for protein retrieval.
            go_index_path (str): Path to the Lucene index for GO term retrieval.
            pmid2text_path (str): Path to the PMID to text mapping file.
            task_pro2go_path (str): Path to the task-specific protein to GO mapping file.
            input_file (str): Path to the input file containing protein-PMID mappings.
            task (str): Task type ('bp', 'mf', 'cc'). Default is 'bp'.
            pro (int): Number of proteins to retrieve. Default is 3.
            filter_rank (bool): Whether to filter by rank. Default is False.
            save_dir (str): Directory to save results. Default is './results'.
            device (str): Device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.
        """
        self.model_path = model_path
        self.pro_index_path = pro_index_path
        self.go_index_path = go_index_path
        self.metadata_path = metadata_path
        self.pmid2text_path = pmid2text_path
        self.task_pro2go_path = task_pro2go_path
        self.input_file = input_file
        self.task = task
        self.pro = pro
        self.filter_rank = filter_rank
        self.save_dir = save_dir
        self.device = device

        # Load resources
        self.metadata = self._load_metadata(self.metadata_path)
        self.pmid2text = self._load_metadata(self.pmid2text_path)
        self.pro_searcher = LuceneSearcher(self.pro_index_path)
        self.go_searcher = LuceneSearcher(self.go_index_path)

        # Initialize models
        self.t5_reranker = self._initialize_t5_model()
        self.cross_encoder = CrossEncoder(self.model_path, max_length=512, device=self.device)

        # Load NLTK tokenizer
        self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def _load_metadata(file_path: str) -> Dict:
        """Load metadata from a .npy file."""
        return np.load(file_path, allow_pickle=True).item()

    @staticmethod
    def _initialize_t5_model():
        """Initialize the T5 reranker model and tokenizer."""
        model = T5ForConditionalGeneration.from_pretrained("castorini/monot5-base-med-msmarco")
        model = nn.DataParallel(model).cuda().module
        tokenizer = MonoT5.get_tokenizer("t5-base")
        return MonoT5(model, tokenizer)

    def data_extract(self) -> Dict[str, str]:
        """
        Extract relevant text for proteins based on PMIDs.

        Returns:
            Dict[str, str]: A dictionary mapping protein IDs to extracted text.
        """
        save_file = os.path.join(self.save_dir, f"{self.task}_extracted_texts.npy")
        if os.path.exists(save_file):
            print(f"Loading extracted data from: {save_file}")
            return self._load_metadata(save_file)

        pro2text = {}
        pros = set()

        with open(self.input_file) as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Extracting data"):
                line = line.split()
                pro = line[0]
                pmid = line[1].strip()

                # Filter by rank if required
                if self.filter_rank:
                    if len(line) < 3:
                        print(f"Missing score for protein {pro}")
                        continue
                    score = float(line[2].strip())
                    if pro not in pros:
                        threshold = min(0.5, score)
                        pros.add(pro)
                    if score < threshold:
                        continue

                # Get protein name
                if pro not in self.metadata:
                    print(f"Missing Protein Name: {pro}")
                    continue

                # Get document content
                if pmid in self.pmid2text:
                    text = self.pmid2text[pmid]
                else:
                    try:
                        text = json.loads(self.pro_searcher.doc(pmid).raw())["contents"]
                    except AttributeError:
                        text = get_text(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}")
                        if text:  # Only cache if text is not empty
                            self.pmid2text[pmid] = text

                sentences = self.tokenizer.tokenize(text)
                if pro not in pro2text:
                    pro2text[pro] = []

                # Filter single-sentence documents
                if len(sentences) <= 1:
                    print(f"Only one sentence in document: {pmid}")
                    continue

                # Filter sentences that are already in pro2text[pro]
                new_sentences = [s for s in sentences if s not in pro2text[pro]]
                pro2text[pro].extend(new_sentences)

        # Save extracted data
        np.save(save_file, pro2text)
        return pro2text

    def all_retrieval_dict(self) -> Dict[str, List[str]]:
        """
        Retrieve GO annotations for proteins.

        Returns:
            Dict[str, List[str]]: A dictionary mapping protein IDs to their associated GO annotations.
        """
        save_file = os.path.join(self.save_dir, f"{self.task}_retrieval_all.npy")
        if os.path.exists(save_file):
            print(f"Loading retrieval data from: {save_file}")
            return self._load_metadata(save_file)

        retrieval_dict = {}
        pro2text = self.data_extract()
        pro2go = self._load_metadata(self.task_pro2go_path)

        for proid, sentences in tqdm(pro2text.items(), desc="Retrieving GO annotations"):
            k = 0
            res = []
            try:
                proname = json.loads(self.pro_searcher.doc(proid).raw())["contents"]
            except AttributeError:
                print(f"Missing Protein Name: {proid}")
                continue

            # Search for related proteins
            results = self.pro_searcher.search(proname, 3000)
            for result in results:
                if k > self.pro + 2:
                    break
                doc = json.loads(result.raw)
                if doc["id"] == proid:
                    continue
                if doc["id"] in pro2go:
                    k += 1
                    res.append(pro2go[doc["id"]])
                else:
                    res.append([])

            if k < self.pro + 2:
                print(f"Insufficient retrievals ({k}) for Protein ID: {proid}")

            retrieval_dict[proid] = res

        np.save(save_file, retrieval_dict)
        return retrieval_dict

    def rerank(self):
        """
        Perform reranking on retrieved GO annotations.
        """
        retrieval_data = self.all_retrieval_dict()
        pro2text = self.data_extract()
        score_dict = os.path.join(self.save_dir, f"{self.task}_t5_scores.npy")

        if os.path.exists(score_dict):
            print(f"Loading scores from cache: {score_dict}")
            score = self._load_metadata(score_dict)
        else:
            score = {}

        data = []

        for proid, go_annotations in tqdm(retrieval_data.items(), desc="Reranking"):
            if len(go_annotations) == 0:
                print(f"No retrievals for Protein ID: {proid}")
                continue

            if proid not in pro2text:
                print(f"Missing text for Protein ID: {proid}")
                continue

            query = f"The protein is \"{self.metadata[proid]}\", the document is \"{pro2text[proid]}\"."
            predicts, goids = [], []

            for goid in go_annotations:
                if len(go_annotations) == 0:
                    continue
                if proid not in score:
                    score[proid] = {}
                if goid in score[proid]:
                    continue

                try:
                    contents = json.loads(self.go_searcher.doc(goid.replace("GO:", "").strip()).raw())["contents"]
                except AttributeError:
                    print(f"GO Missing: {goid}")
                    continue

                goids.append(goid)
                predicts.append([query, contents])

            if len(predicts) == 0:
                continue

            scores = self.cross_encoder.predict(predicts, batch_size=96, show_progress_bar=False)
            for i, goid in enumerate(goids):
                score[proid][goid] = "%.3f" % float(scores[i])

        np.save(score_dict, score)

        for proid, go_annotations in retrieval_data.items():
            res = {goid: str(score[proid].get(goid, 0)) for goid in go_annotations}
            res = sorted(res.items(), key=lambda x: x[1], reverse=True)[:50]
            for goid, s in res:
                data.append(f"{proid}\t{goid}\t{s}\n")

        save_file = os.path.join(self.save_dir, f"{self.task}_rerank.txt")
        with open(save_file, "w") as wf:
            wf.writelines(data)
        print(f"Rerank results saved to: {save_file}")