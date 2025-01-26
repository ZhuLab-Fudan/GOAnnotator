# experiments/predict.py
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pubretriever import pubretriever
from src.goretriever_plus import goretriever_plus


retriever = pubretriever(
    model_path="/path/to/your/model",         
    index_path="/path/to/lucene/index",      
    task="mf",                               
    gpu_id="0"                               
)

# 定义蛋白质 ID 列表
protein_ids = ["P12345", "Q67890"]  

# 定义输出结果路径
output_path = "retrieval_results.map"

# 执行检索与重排序
retriever.process_proteins(
    protein_ids=protein_ids,                 
    output_path=output_path,                 
    metadata=protein_metadata                
)

print(f"Results saved to {output_path}")

retriever = goretriever_plus(
    model_path="/path/to/cross_encoder_model",
    pro_index_path="/path/to/protein_index",
    go_index_path="/path/to/go_index",
    proid2name_path="/path/to/proid2name.npy",
    pmid2text_path="/path/to/pmid2text.npy",
    task_pro2go_path="/path/to/task_pro2go.npy",
    input_file="/path/to/input_file.txt",
    task="bp",
    pro=3,
    filter_rank=False,
    save_dir="./results",
    device="cuda",
)

# Extract data
pro2text = retriever.data_extract()

# Retrieve GO annotations
retrieval_data = retriever.all_retrieval_dict()

# Rerank GO annotations
retriever.rerank()