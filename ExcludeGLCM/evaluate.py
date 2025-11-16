import torch
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from model import SaigenCNN
from balanced_data_loader import create_balanced_data_loaders

def evaluate(config):
    """
    学習済みモデルをテストデータで評価し、混同行列を作成する関数
    """
    device = torch.device("cpu")
    print(f"Using device: {device}")

    _, _, test_loader = create_balanced_data_loaders(config)

    model = SaigenCNN(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)

    # 3. 保存された学習済みパラメータを読み込みます
    model_path = config['training']['model_save_path']
    try:
        # PyTorchの警告を避けるため、weights_only=Trueを追加
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nLoaded best model from: {model_path}")
    except FileNotFoundError:
        print(f"エラー: モデルファイル '{model_path}' が見つかりません。パスを確認してください。")
        return
    except RuntimeError as e:
        print(f"エラー: モデルの読み込みに失敗しました。モデルの構造と設定ファイルの内容が、学習時と完全に一致しているか確認してください。")
        print(e)
        return

    # 4. 評価を実行
    model.eval()
    all_preds = []
    all_labels = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. 結果を表示し、グラフを保存
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"\n✅ Final Test Accuracy: {accuracy * 100:.2f}%")

    class_names = ['Original', 'Tampered']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Test Set')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    plot_path = 'confusion_matrix.png'
    plt.savefig(plot_path)
    print(f"📈 Confusion matrix plot saved to {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    evaluate(config)