import argparse
from src.goretriever_plus.goretriever_plus import GORetrieverPlus
from src.pubretriever.pubretriever import PubRetriever
from utils import Config

def main():
    parser = argparse.ArgumentParser(description="Run GORetrieverPlus")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save results")
    parser.add_argument('--task', type=str, required=True, help="Task name")
    parser.add_argument('--data', type=str, required=True, help="Data file")
    parser.add_argument('--input', type=str, required=True, help="Input file")
    parser.add_argument('--pro_num', type=int, default=3, help="Number of proteins to consider")
    parser.add_argument('--filter_rank', type=bool, default=True, help="Whether to filter rank")


    args = parser.parse_args()


    config = Config()
    retriever = GORetrieverPlus(config, args.save_dir, args.task, args.data, args.input, args.pro_num, args.filter_rank)
    
    retriever.rerank()

if __name__ == "__main__":
    main()