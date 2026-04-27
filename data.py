import os
import random
import torch
import pandas as pd
from typing import List, Tuple, Dict
from torch.utils.data import Dataset

def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame,pd.DataFrame]:
    """读取interactions.csv和items.csv，返回(interactions_df, items_df)"""
    if not os.path.isabs(data_dir):
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        data_dir = os.path.join(project_root, "data")

    interactions4train_path = os.path.join(data_dir, "interactions_train.csv")
    interactions4val_path = os.path.join(data_dir, "interactions_val.csv")
    items_path = os.path.join(data_dir, "items.csv")
    
    interactions_df = pd.read_csv(interactions4train_path)
    interactions4val_df = pd.read_csv(interactions4val_path)
    items_df = pd.read_csv(items_path)
    return interactions_df, items_df, interactions4val_df

class DualTowerDataset(Dataset):
    def __init__(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, tokenizer, 
                 max_length=128, num_negatives=4, max_history=10):
        self.interactions = interactions_df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.max_history = max_history

        self.item_texts = {}
        self.item_titles = {}
        for _, row in items_df.iterrows():
            title = str(row['title'])
            desc = str(row['description'])
            self.item_texts[row['item_id']] = f"{title}[SEP]{desc}"
            self.item_titles[row['item_id']] = title
            
        self.all_item_ids = items_df['item_id'].tolist()

        self.histories = []
        user_hist_map = {}
        for _, row in self.interactions.iterrows():
            uid = row['user_id']
            iid = row['item_id']
            current_hist = user_hist_map.get(uid, [])[-self.max_history:]
            self.histories.append([self.item_titles.get(i, "") for i in current_hist])
            if uid not in user_hist_map:
                user_hist_map[uid] = []
            user_hist_map[uid].append(iid)

    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.interactions.iloc[idx]
        pos_item_id = row['item_id']

        user_history_titles = self.histories[idx]
        user_text = "[SEP]".join(user_history_titles) if user_history_titles else "无历史"
        pos_item_text = self.item_texts[pos_item_id]
        
        # 随机负采样
        neg_item_ids = []
        while len(neg_item_ids) < self.num_negatives:
            neg_id = random.choice(self.all_item_ids)
            if neg_id != pos_item_id:
                neg_item_ids.append(neg_id)
                
        neg_item_texts = [self.item_texts[nid] for nid in neg_item_ids]
        
        # Tokenize
        user_enc = self.tokenizer(user_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        pos_enc = self.tokenizer(pos_item_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        neg_enc = self.tokenizer(neg_item_texts, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        
        return {
            "user_input_ids": user_enc['input_ids'].squeeze(0),
            "user_attention_mask": user_enc['attention_mask'].squeeze(0),
            "pos_item_input_ids": pos_enc['input_ids'].squeeze(0),
            "pos_item_attention_mask": pos_enc['attention_mask'].squeeze(0),
            "neg_item_input_ids": neg_enc['input_ids'],           # (num_negatives, max_length)
            "neg_item_attention_mask": neg_enc['attention_mask']  # (num_negatives, max_length)
        }

def collate_fn(batch: List[dict]) -> dict:
    """批量化张量"""
    return {
        "user_input_ids": torch.stack([x["user_input_ids"] for x in batch]),
        "user_attention_mask": torch.stack([x["user_attention_mask"] for x in batch]),
        "pos_item_input_ids": torch.stack([x["pos_item_input_ids"] for x in batch]),
        "pos_item_attention_mask": torch.stack([x["pos_item_attention_mask"] for x in batch]),
        "neg_item_input_ids": torch.stack([x["neg_item_input_ids"] for x in batch]),
        "neg_item_attention_mask": torch.stack([x["neg_item_attention_mask"] for x in batch]),
    }