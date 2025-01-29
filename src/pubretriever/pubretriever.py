import os
import json
from typing import List, Dict
from sentence_transformers import CrossEncoder
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import numpy as np

# Import utilities
from utils import average_scores
from src.utils import TASK_DEFINITIONS, filter_text, filter_species, extract_for_FIR, extract_for_train, Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PubRetriever:
    """
    A class for protein-related literature retrieval and reranking for function annotation.
    """

    def __init__(self, config: Config, model_path: str, input: List[str], output: List[str], k: int = 50, task: str = "mf"):
        """
        Initialize the PubRetriever.
        
        Args:
            model_path (str): Path to the pre-trained CrossEncoder model.
            input (List[str]): List of protein IDs to process.
            k (int): Number of documents to retrieve. Default is 50.
            task (str): Annotation task type ('mf', 'bp', 'cc'). Default is 'mf'.
        """
        self.model = CrossEncoder(model_path, max_length=512)
        self.searcher = LuceneSearcher(config.PMID_INDEX_PATH)
        self.metadata = np.load(config.METADATA_PATH, allow_pickle=True).item()
        self.input = input  
        self.output = output
        self.task = task
        self.k = k
        
    def full_information_retrieval(self, protein_metadata: Dict) -> List[Dict[str, str]]:
        """
        Perform full information retrieval (FIR) for a given protein.
        
        Args:
            protein_metadata (Dict): Metadata of the protein, including names and descriptions.
        
        Returns:
            List[Dict[str, str]]: A list of retrieved documents with PMID and text content.
        """
        protein_name = extract_for_FIR(protein_metadata).replace('gene', '').replace('protein', '')
        query = f"function or role of {protein_name}"
        results = self.searcher.search(query, self.k)
        pmids = []
        for result in results:
            pmid = json.loads(result.raw)
            pmids.append({
                'pmid': pmid['id'],
                'text': pmid['contents']
            })
        return pmids
        

    def meta_FGR(self, query: str, weight: float) -> Dict[str, List[float]]:
        """
        Perform fine-grained retrieval (FGR) for a given query and update scores.
        
        Args:
            query (str): The search query.
            weight (float): Weight to scale the scores.
        
        Returns:
            Dict[str, List[float]]: A dictionary of document IDs and their scores.
        """
        scores = {}
        results = self.searcher.search(query, self.k)
        for result in results:
            pmid = json.loads(result.raw)['id']
            scores.setdefault(pmid, []).append(weight * result.score)
        return scores
    
    def fine_grained_retrieval(self, protein_metadata: Dict) -> List[Dict[str, str]]:
        """
        Perform fine-grained retrieval (FGR) for a given protein using its metadata.
        
        Args:
            protein_metadata (Dict): Metadata of the protein, including names and descriptions.
        
        Returns:
            List[Dict[str, str]]: A list of retrieved documents with PMID and text content.
        """
        scores = {}
        names = protein_metadata.get("gene name", {}).get("name", [])
        if len(names) > 0:
            for item in names:
                if item.get('Name'):
                    query = filter_text(item['Name'])
                    scores = self.meta_FGR(query, 1/len(query.split(' ')))
                    if len(scores) >= self.k:
                        query = (filter_text(item['Name']) + ' ') * 4 + ' ' + filter_species(protein_metadata['species'])
                        scores = self.meta_FGR(query, 1/len(query.split(' ')))
                elif item.get('Synonyms') and item['Synonyms'] != 'N/A':
                    query = filter_text(item['Synonyms'])
                    scores = self.meta_FGR(query, 1/len(query.split(' ')))
                    if len(scores) >= self.k:
                        query = (filter_text(item['Synonyms']) + ' ') * 4 + ' ' + filter_species(protein_metadata['species'])
                        scores = self.meta_FGR(query, 1/len(query.split(' ')))
        
        if protein_metadata.get("short name"):
            if isinstance(protein_metadata["short name"], str):
                query = filter_text(protein_metadata["short name"])
                scores = self.meta_FGR(query, 1/len(query.split(' ')))
            elif isinstance(protein_metadata["short name"], list):
                for item in protein_metadata["short name"]:
                    query = filter_text(item)
                    scores = self.meta_FGR(query, 1/len(query.split(' ')))  
                    
        if len(scores) >= self.k:
            for pmid in scores:
                scores[pmid] = average_scores(scores[pmid])
            pmids = sorted(scores, key=lambda x: scores[x], reverse=True)[:self.k]
            res = []
            for pmid in pmids:
                pmid = json.loads(self.searcher.doc(pmid).raw())
                res.append({
                    'pmid': pmid['id'],
                    'text': pmid['contents']
                })
            return res
        
        if protein_metadata.get('full name'):
            if protein_metadata['full name'].get('RecName'):
                query = filter_text(protein_metadata['full name']['RecName']) 
                scores = self.meta_FGR(query, 1/len(query.split(' ')))
            if protein_metadata['full name'].get('AltName'):
                for item in protein_metadata['full name']['AltName'].split(','):
                    query = filter_text(item) 
                    scores = self.meta_FGR(query, 1/len(query.split(' ')))
        
        for pmid in scores:
            try:
                scores[pmid] = average_scores(scores[pmid])
            except TypeError:
                print(scores, protein_metadata)
        
        pmids = sorted(scores, key=lambda x: scores[x], reverse=True)[:self.k]
        res = []
        for pmid in pmids:
            pmid = json.loads(self.searcher.doc(pmid).raw())
            res.append({
                'pmid': pmid['id'],
                'text': pmid['contents']
            })
        return res

    def hybrid_retrieval(self, protein_metadata: Dict) -> List[Dict[str, str]]:
        """
        Perform hybrid retrieval by combining results from FIR and FGR.
        
        Args:
            protein_metadata (Dict): Metadata of the protein, including names and descriptions.
        
        Returns:
            List[Dict[str, str]]: A list of retrieved documents with PMID and text content.
        """
        FIR_results = self.full_information_retrieval(protein_metadata)
        FGR_results = self.fine_grained_retrieval(protein_metadata)
        merged_dict = {item['pmid']: item for item in FIR_results + FGR_results}
        return list(merged_dict.values())
        

    def rerank_documents(self, protein_metadata: Dict, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Rerank retrieved documents using the CrossEncoder model.
        
        Args:
            protein_metadata (Dict): Metadata of the protein, including names and descriptions.
            documents (List[Dict[str, str]]): A list of retrieved documents with content.
        
        Returns:
            List[Dict[str, str]]: A sorted list of documents with reranked scores.
        """
        protein_name = extract_for_train(protein_metadata)
        query = f"{protein_name} The {TASK_DEFINITIONS[self.task]} is "
        inputs = [[query, doc['text']] for doc in documents]
        scores = self.model.predict(inputs, batch_size=96, show_progress_bar=False)
        for i, doc in enumerate(documents):
            doc['score'] = scores[i]
        return sorted(documents, key=lambda x: x['score'], reverse=True)
    
    def retrieve_and_rerank(self, protein_metadata: Dict) -> List[Dict[str, str]]:
        """
        Retrieve and rerank documents for a given protein.
        
        Args:
            protein_metadata (Dict): Metadata of the protein, including names and descriptions.
        
        Returns:
            List[Dict[str, str]]: Top reranked documents.
        """
        hybrid_retrieval_results = self.hybrid_retrieval(protein_metadata)
        return self.rerank_documents(protein_metadata, hybrid_retrieval_results)

    def process_proteins(self):
        """
        Process a list of protein IDs, perform retrieval and reranking, and save results.
        
        Args:
            output_path (str): Path to save the results.
        """
        with open(self.input) as input_file:
            with open(self.output, "w") as output_file:
                proids = [_.strip() for _ in input_file.readlines]
                for proid in tqdm(proids, desc="Processing proteins"):
                    if proid not in self.metadata:
                        print(f"Metadata missing for Protein ID: {proid}")
                        continue
                    protein_metadata = self.metadata.get(proid, {})
                    documents = self.retrieve_and_rerank(protein_metadata)
                    for doc in documents:
                        output_file.write(f"{proid}\t{doc['pmid']}\t{doc['score']:.4f}\n")