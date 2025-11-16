import torch
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import pandas as pd

from model import SaigenCNN
from balanced_data_loader import create_balanced_data_loaders

def copy_image_to_result(image_path, dest_dir, prediction, confidence, true_label):
    """
    画像を適切な結果ディレクトリにコピー
    """
    if not os.path.exists(image_path):
        print(f"⚠️ 画像が見つかりません: {image_path}")
        return
    
    # ファイル名に予測情報を追加
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    labels = ['original', 'tampered']
    new_filename = f"{name}_pred{labels[prediction]}_true{labels[true_label]}_conf{confidence:.3f}{ext}"
    
    dest_path = os.path.join(dest_dir, new_filename)
    shutil.copy2(image_path, dest_path)

def classify_evaluate(config):
    """
    テストデータを評価し、結果別に画像を分類保存
    """
    print(f"🚀 GLCMなし軽量CNN - テストデータ分類評価")
    print(f"=" * 60)
    
    
    device = torch.device("cpu")
    print(f"🖥️ 使用デバイス: {device}")

    # データローダー取得
    _, _, test_loader = create_balanced_data_loaders(config)
    print(f"📊 テストデータ: {len(test_loader.dataset)} 枚")

    # モデル読み込み
    model = SaigenCNN(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)

    model_path = config['training']['model_save_path']
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ モデル読み込み完了: {model_path}")
    except FileNotFoundError:
        print(f"❌ エラー: モデルファイル '{model_path}' が見つかりません。")
        return
    except RuntimeError as e:
        print(f"❌ エラー: モデルの読み込みに失敗しました。")
        print(e)
        return

    # 評価実行
    model.eval()
    all_results = []
    
    print(f"\n🔍 テストデータ評価中...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="分類評価")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # 確率と予測計算
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            # バッチ内の各画像について処理
            for i in range(inputs.size(0)):
                true_label = labels[i].item()
                predicted_label = predictions[i].item()
                confidence = confidences[i].item()
                
                # 画像インデックス計算（実際の画像パス取得のため）
                image_idx = batch_idx * test_loader.batch_size + i
                
                # データセットから実際のファイルパスを取得
                try:
                    # test_loaderのdatasetから画像パスを取得
                    if hasattr(test_loader.dataset, 'image_paths'):
                        image_path = test_loader.dataset.image_paths[image_idx]
                    elif hasattr(test_loader.dataset, 'samples'):
                        image_path = test_loader.dataset.samples[image_idx][0]
                    else:
                        # フォールバック: インデックスベースの仮想パス
                        image_path = f"test_image_{image_idx:04d}.jpg"
                        print(f"⚠️ 実際のパスが取得できません。仮想パス使用: {image_path}")
                        continue
                        
                except IndexError:
                    print(f"⚠️ インデックス {image_idx} の画像パスが取得できません")
                    continue
                
                # 結果記録
                result = {
                    'image_path': image_path,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'correct': true_label == predicted_label
                }
                all_results.append(result)
                
                # 分類に基づいて画像をコピー
                if true_label == predicted_label:  # 正解
                    if true_label == 0:  # 正しくオリジナル判定
                        dest_dir = "result/correct/correct_original"
                    else:  # 正しく改ざん判定
                        dest_dir = "result/correct/correct_tampered"
                else:  # 不正解
                    if predicted_label == 1:  # オリジナル→改ざん誤判定
                        dest_dir = "result/incorrect/false_positive"
                    else:  # 改ざん→オリジナル誤判定
                        dest_dir = "result/incorrect/false_negative"
                
                # 画像コピー実行
                copy_image_to_result(image_path, dest_dir, predicted_label, confidence, true_label)

    # 結果分析
    df_results = pd.DataFrame(all_results)
    
    # 統計計算
    total_images = len(all_results)
    correct_predictions = len(df_results[df_results['correct'] == True])
    accuracy = correct_predictions / total_images if total_images > 0 else 0
    
    # 詳細統計
    correct_original = len(df_results[(df_results['true_label'] == 0) & (df_results['predicted_label'] == 0)])
    correct_tampered = len(df_results[(df_results['true_label'] == 1) & (df_results['predicted_label'] == 1)])
    false_positive = len(df_results[(df_results['true_label'] == 0) & (df_results['predicted_label'] == 1)])
    false_negative = len(df_results[(df_results['true_label'] == 1) & (df_results['predicted_label'] == 0)])
    
    print(f"\n📊 分類結果統計:")
    print(f"   総画像数: {total_images}")
    print(f"   総合精度: {accuracy * 100:.2f}%")
    print(f"")
    print(f"   ✅ 正解分類:")
    print(f"      正しくオリジナル判定: {correct_original} 枚")
    print(f"      正しく改ざん判定: {correct_tampered} 枚")
    print(f"   ❌ 不正解分類:")
    print(f"      誤改ざん判定 (False Positive): {false_positive} 枚")
    print(f"      改ざん見逃し (False Negative): {false_negative} 枚")
    
    # ディレクトリ別ファイル数確認
    print(f"\n📁 結果ディレクトリ内容:")
    result_dirs = {
        "correct/correct_original": correct_original,
        "correct/correct_tampered": correct_tampered,
        "incorrect/false_positive": false_positive,
        "incorrect/false_negative": false_negative
    }
    
    for dir_name, expected_count in result_dirs.items():
        full_path = f"result/{dir_name}"
        actual_count = len([f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
        status = "✅" if actual_count == expected_count else "⚠️"
        print(f"   {status} {dir_name}: {actual_count} 枚")
    
    # 詳細結果をCSVで保存
    csv_path = "result/detailed_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 詳細結果を保存: {csv_path}")
    
    # 混同行列も保存
    all_labels = [r['true_label'] for r in all_results]
    all_preds = [r['predicted_label'] for r in all_results]
    
    class_names = ['Original', 'Tampered']
    print(f"\n📋 分類レポート:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '画像数'})
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    plot_path = 'result/confusion_matrix_classification.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📈 混同行列を保存: {plot_path}")
    
    # 信頼度分析
    print(f"\n🔍 信頼度分析:")
    confidences = [r['confidence'] for r in all_results]
    correct_confidences = [r['confidence'] for r in all_results if r['correct']]
    incorrect_confidences = [r['confidence'] for r in all_results if not r['correct']]
    
    print(f"   全体平均信頼度: {np.mean(confidences):.3f}")
    print(f"   正解時平均信頼度: {np.mean(correct_confidences):.3f}")
    print(f"   不正解時平均信頼度: {np.mean(incorrect_confidences):.3f}")
    
    return df_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="テストデータの分類評価と画像分類")
    parser.add_argument('--config', type=str, required=True, help="設定ファイルパス (config.yaml)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    results = classify_evaluate(config)