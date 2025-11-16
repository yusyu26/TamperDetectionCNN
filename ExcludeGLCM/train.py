import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm
import random
import numpy as np

# 作成したモデルをインポート
from model import SaigenCNN 


from balanced_data_loader import create_balanced_data_loaders

def set_seed(seed):
    """
    完全な再現性を確保するためのシード設定
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # マルチGPU用
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # MPSの場合の設定
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def train(config):
    # シード値の設定（再現性のため）
    seed = config['dataset']['seed']
    set_seed(seed)
    print(f"🌱 Seed set to: {seed}")
    
    # デバイスの設定 (Apple SiliconのGPU(MPS)またはCPUを自動選択)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # データローダーの作成
    train_loader, val_loader, _ = create_balanced_data_loaders(config)
    # モデルの初期化
    model = SaigenCNN(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)

    # 損失関数とオプティマイザの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    best_val_accuracy = 0.0

    print("Training started...")
    # 学習ループ
    for epoch in range(config['training']['epochs']):
        # --- 学習フェーズ ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # tqdmで進捗バーを表示
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_pbar.set_postfix({'loss': loss.item()})

        train_accuracy = 100 * correct_train / total_train
        
        # --- 検証フェーズ ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_pbar.set_postfix({'loss': loss.item()})
        
        val_accuracy = 100 * correct_val / total_val

        print(f"Epoch {epoch+1}/{config['training']['epochs']}:")
        print(f"  Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%")

        # 最高の検証精度が出たモデルを保存
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), config['training']['model_save_path'])
            print(f"✨ New best model saved with validation accuracy: {best_val_accuracy:.2f}%")

    print("Training finished!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN model for image tamper detection.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file.")
    args = parser.parse_args()

    # 設定ファイルを読み込む
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config)