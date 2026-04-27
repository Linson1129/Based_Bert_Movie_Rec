import json
import torch
import faiss
import numpy as np
import argparse
from transformers import BertTokenizer
from model import DualTowerModel

def recommend(user_text: str, top_k: int = 10, 
              checkpoint_path: str = "checkpoints/final.pt",
              item_embeddings_path: str = "checkpoints/item_embeddings.npy",
              item_ids_path: str = "checkpoints/item_ids.json") -> list:
    
    # 加载模型配置 (写死默认配置避免额外依赖)
    bert_model_name = "bert-base-chinese"
    max_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = DualTowerModel(bert_model_name=bert_model_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 加载向量与ID映射
    item_embeddings = np.load(item_embeddings_path)
    with open(item_ids_path, 'r') as f:
        item_ids = json.load(f)
        
    # 构建Faiss
    d = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(item_embeddings)
    
    # 推理
    with torch.no_grad():
        enc = tokenizer(user_text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt').to(device)
        u_vec = model.encode(enc['input_ids'], enc['attention_mask']).cpu().numpy()
        faiss.normalize_L2(u_vec)
        
        scores, I = index.search(u_vec, top_k)
        
    results = []
    for score, idx in zip(scores[0], I[0]):
        results.append((item_ids[idx], float(score)))
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_text", type=str, required=True, help="用户历史拼接文本")
    parser.add_argument("--top_k", type=int, default=10, help="返回Top-K结果")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt")
    args = parser.parse_args()
    
    results = recommend(args.user_text, args.top_k, args.checkpoint)
    print(f"\nTop-{args.top_k} Recommendations for user history: '{args.user_text}'")
    for rank, (item_id, score) in enumerate(results, 1):
        print(f"{rank}. Item ID: {item_id} | Similarity: {score:.4f}")