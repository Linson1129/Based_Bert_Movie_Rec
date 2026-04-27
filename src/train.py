import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from data import load_data, DualTowerDataset, collate_fn
from model import DualTowerModel

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(config_path: str):
    # --- 新增：自动路径修正 ---
    if not os.path.isabs(config_path):
        # 获取当前执行脚本 src/train.py 的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 它的上一级才是项目根目录 Rec
        project_root = os.path.dirname(script_dir)
        # 重新拼接出正确的绝对路径
        config_path = os.path.join(project_root, config_path)
    # -----------------------

    print(f"正在尝试读取配置文件: {config_path}")

    # 1. 读取 configs/default.yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 2. 设置随机种子
    set_seed(config['seed'])
    
    # 3. 设置设备
    device = torch.device(config['device'])
    
    # 4. 加载 tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])
    
    # 5. 读取数据
    train_df, items_df, test_df = load_data(config['data']['data_dir'])

    # 6. 构建 Dataset
    dataset = DualTowerDataset(
        interactions_df=train_df, 
        items_df=items_df, 
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        num_negatives=config['data']['num_negatives'],
        max_history=config['data']['max_history']
    )
    
    # 7. 构建 DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0
    )
    
    # 8. 初始化 Model
    model = DualTowerModel(
        bert_model_name=config['model']['bert_model_name'],
        embedding_dim=config['model']['embedding_dim'],
        temperature=config['model']['temperature'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # 9. 优化器
    optimizer = AdamW(model.parameters(), lr=float(config['train']['lr']), weight_decay=config['train']['weight_decay'])
    
    # 10. 学习率调度器
    total_steps = len(dataloader) * config['train']['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * config['train']['warmup_ratio']),
        num_training_steps=total_steps
    )
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # 11. 训练循环
    model.train()
    for epoch in range(config['train']['epochs']):
        total_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{config['train']['epochs']} - Loss: {total_loss/len(dataloader):.4f}")
        
        if (epoch + 1) % config['train']['save_every'] == 0:
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pt")
            
    # 12. 保存最终模型
    torch.save(model.state_dict(), "checkpoints/final.pt")
    print("Training complete. Model saved to checkpoints/final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    # --- 新增：自动路径转换逻辑 ---
    config_path = args.config
    if not os.path.isabs(config_path):
        # 获取当前 train.py 的绝对路径 (D:\Working\VScode\Rec\src\train.py)
        current_file_path = os.path.abspath(__file__)
        # 获取项目根目录 (D:\Working\VScode\Rec)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        # 拼接出真正的配置文件路径
        config_path = os.path.join(project_root, args.config)
    # ---------------------------

    if not os.path.exists(config_path):
        print(f"仍然找不到文件，请检查该路径是否存在: {config_path}")
    else:
        train(config_path) # 传入修正后的绝对路径


