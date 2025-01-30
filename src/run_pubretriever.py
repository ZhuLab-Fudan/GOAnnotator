import argparse
from pubretriever.pubretriever import PubRetriever
from utils import Config


def main():
    parser = argparse.ArgumentParser(description="Run PubRetriever")
    parser.add_argument('--task', type=str, required=True, help="Task name")
    parser.add_argument('--input', type=str, required=True, help="Comma-separated list of protein IDs to process")
    parser.add_argument('--output', type=str, required=True, help="Output Path")
    parser.add_argument('--model_path', type=str, default="whitneyyan0122/pubretriever-reranker", help="Path to the pre-trained model")
    parser.add_argument('--k', type=int, default=50, help="Number of documents to retrieve")


    args = parser.parse_args()
    
    config = Config()
    retriever = PubRetriever(config, args.model_path, args.input, args.output , args.k, args.task)
    retriever.process_proteins()
    
if __name__ == "__main__":
    main()