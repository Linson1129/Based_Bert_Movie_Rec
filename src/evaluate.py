import os
import json
import yaml
import torch
import faiss
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import BertTokenizer

from data import load_data
from model import DualTowerModel

def evaluate(checkpoint_path: str, config_path: str):
    # 1. 加载配置和模型
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['device'])
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])
    
    model = DualTowerModel(
        bert_model_name=config['model']['bert_model_name'],
        embedding_dim=config['model']['embedding_dim'],
        temperature=config['model']['temperature'],
        dropout=config['model']['dropout']
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    train_df, items_df, test_df = load_data(config['data']['data_dir'])

    user_train_history = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    train_df_sorted = train_df.sort_values(['user_id', 'timestamp'])
    item_titles = dict(zip(items_df['item_id'], items_df['title']))

    print("Building Item Embeddings...")
    item_ids = items_df['item_id'].tolist()
    item_embeddings = []
    
    with torch.no_grad():
        for _, row in tqdm(items_df.iterrows(), total=len(items_df)):
            text = f"{row['title']}[SEP]{row['description']}"
            enc = tokenizer(text, max_length=config['data']['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(device)
            vec = model.encode(enc['input_ids'], enc['attention_mask']).cpu().numpy()
            item_embeddings.append(vec[0])
            
    item_embeddings = np.array(item_embeddings, dtype=np.float32)
    faiss.normalize_L2(item_embeddings)
    
    os.makedirs("checkpoints", exist_ok=True)
    np.save("checkpoints/item_embeddings.npy", item_embeddings)
    with open("checkpoints/item_ids.json", "w") as f:
        json.dump(item_ids, f)

    d = config['model']['embedding_dim']
    if len(item_ids) < 100000:
        index = faiss.IndexFlatIP(d)
    else:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_INNER_PRODUCT)
        index.train(item_embeddings)
        
    index.add(item_embeddings)

    print("Evaluating...")
    metrics = {f"Recall@{k}": [] for k in config['eval']['top_k']}
    metrics.update({f"NDCG@{k}": [] for k in config['eval']['top_k']})
    
    max_k = max(config['eval']['top_k'])
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            uid = row['user_id']
            target_iid = row['item_id']

            hist_items = train_df_sorted[train_df_sorted['user_id'] == uid]['item_id'].tolist()[-config['data']['max_history']:]
            hist_titles = [str(item_titles.get(i, "")) for i in hist_items]
            user_text = "[SEP]".join(hist_titles) if hist_titles else "无历史"
            
            enc = tokenizer(user_text, max_length=config['data']['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(device)
            u_vec = model.encode(enc['input_ids'], enc['attention_mask']).cpu().numpy()
            faiss.normalize_L2(u_vec)

            search_k = max_k + len(user_train_history.get(uid, set()))
            scores, I = index.search(u_vec, search_k)

            retrieved_items = [item_ids[idx] for idx in I[0]]
            train_items = user_train_history.get(uid, set())
            valid_retrieved = [i for i in retrieved_items if i not in train_items][:max_k]

            for k in config['eval']['top_k']:
                top_k_items = valid_retrieved[:k]
                hit = 1.0 if target_iid in top_k_items else 0.0
                metrics[f"Recall@{k}"].append(hit)
                
                if hit > 0:
                    rank = top_k_items.index(target_iid)
                    ndcg = 1.0 / np.log2(rank + 2)
                else:
                    ndcg = 0.0
                metrics[f"NDCG@{k}"].append(ndcg)

    result = {k: np.mean(v) for k, v in metrics.items()}
    print("\nEvaluation Results:")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config)