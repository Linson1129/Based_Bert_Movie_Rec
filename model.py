import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class DualTowerModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', 
                 embedding_dim=768, temperature=0.07, dropout=0.1):
        super(DualTowerModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(p=dropout)
        self.temperature = temperature
        
    def encode(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]
        return self.dropout(cls_vec)
    
    def forward(self, user_input_ids, user_attention_mask,
                pos_item_input_ids, pos_item_attention_mask,
                neg_item_input_ids, neg_item_attention_mask) -> torch.Tensor:
        
        batch_size = user_input_ids.size(0)
        num_negatives = neg_item_input_ids.size(1)
        max_length = neg_item_input_ids.size(2)
        
        user_vec = self.encode(user_input_ids, user_attention_mask)
        pos_vec = self.encode(pos_item_input_ids, pos_item_attention_mask)
        
        neg_vec = self.encode(
            neg_item_input_ids.view(-1, max_length), 
            neg_item_attention_mask.view(-1, max_length)
        )
        neg_vec = neg_vec.view(batch_size, num_negatives, -1)
        
        pos_sim = (user_vec * pos_vec).sum(dim=-1) / self.temperature
        neg_sim = torch.bmm(neg_vec, user_vec.unsqueeze(-1)).squeeze(-1) / self.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_input_ids.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss