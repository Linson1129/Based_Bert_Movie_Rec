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
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, config_path)

    print(f"正在尝试读取配置文件: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = torch.device(config['device'])
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])
    train_df, items_df, test_df = load_data(config['data']['data_dir'])

    dataset = DualTowerDataset(
        interactions_df=train_df, 
        items_df=items_df, 
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        num_negatives=config['data']['num_negatives'],
        max_history=config['data']['max_history']
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0
    )

    model = DualTowerModel(
        bert_model_name=config['model']['bert_model_name'],
        embedding_dim=config['model']['embedding_dim'],
        temperature=config['model']['temperature'],
        dropout=config['model']['dropout']
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(config['train']['lr']), weight_decay=config['train']['weight_decay'])
    total_steps = len(dataloader) * config['train']['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * config['train']['warmup_ratio']),
        num_training_steps=total_steps
    )

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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "final.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ 模型已成功保存至绝对路径: {checkpoint_path}")
    print("Training complete. Model saved to checkpoints/final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        config_path = os.path.join(project_root, args.config)

    if not os.path.exists(config_path):
        print(f"仍然找不到文件，请检查该路径是否存在: {config_path}")
    else:
        train(config_path)


