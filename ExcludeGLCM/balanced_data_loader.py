#!/usr/bin/env python3
"""
バランス調整済みデータローダー
改ざん画像の枚数に合わせてオリジナル画像を選択し、各クラス同じ枚数でバランスを取る
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import random
import yaml
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from utils_90percent import create_7_channel_features, normalize_image


class BalancedSaigen90Dataset(Dataset):
    """バランス調整済みSaigen90Dataset"""
    
    def __init__(self, image_paths: List[str], labels: List[int], config: Dict[str, Any], mode: str = 'train'):
        """
        Args:
            image_paths: 画像ファイルパスのリスト
            labels: ラベルのリスト（0: オリジナル, 1: 改ざん）
            config: 設定辞書
            mode: 'train', 'val', 'test'のいずれか
        """
        self.image_paths = image_paths
        self.labels = labels
        self.config = config
        self.mode = mode
        
        # 設定抽出
        self.image_size = tuple(config['dataset']['image_size'])
        self.preprocessing_config = config['preprocessing']
        
        print(f"📊 BalancedDataset[{mode}]: {len(self.image_paths)} samples")
        print(f"   オリジナル: {labels.count(0)} / 改ざん: {labels.count(1)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        データ取得
        
        Returns:
            features: 23チャンネル特徴量 (23, H, W)
            label: ラベル (scalar)
        """
        # 画像読み込み
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            raise RuntimeError(f"画像読み込み失敗: {image_path}")
        
        # BGR → RGB変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # リサイズ
        image = cv2.resize(image, self.image_size)
        
        # データ拡張（訓練時のみ適用）
        if self.mode == 'train':
            image = self._apply_data_augmentation(image)
        # 7チャンネル特徴量抽出
        features = create_7_channel_features(image, self.config)
        
        # 正規化
        if self.preprocessing_config.get('normalize', True):
            features = normalize_image(features)
        
        # PyTorchテンソルに変換 (H, W, C) → (C, H, W)
        features = torch.from_numpy(features).float().permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return features, label
    
    def _apply_data_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        データ拡張の適用（改ざんパターンを考慮して控えめ）
        
        Args:
            image: RGB画像 (H, W, 3)
        
        Returns:
            拡張済み画像 (H, W, 3)
        """
        # 設定から拡張パラメータを取得
        augmentation_config = self.config.get('augmentation', {})
        
        # 水平反転
        horizontal_flip = augmentation_config.get('horizontal_flip', 0)
        if horizontal_flip > 0 and random.random() < horizontal_flip:
            image = cv2.flip(image, 1)
        
        # 回転
        rotation_range = augmentation_config.get('rotation_range', 0)
        if rotation_range > 0:
            angle = random.uniform(-rotation_range, rotation_range)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)
        
        # 明度調整
        brightness_range = augmentation_config.get('brightness_range', 0)
        if brightness_range > 0:
            brightness_factor = 1.0 + random.uniform(-brightness_range, brightness_range)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        # コントラスト調整
        contrast_range = augmentation_config.get('contrast_range', 0)
        if contrast_range > 0:
            contrast_factor = 1.0 + random.uniform(-contrast_range, contrast_range)
            image = np.clip((image - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        return image


def load_balanced_dataset_paths(config: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    """
    バランス調整済みデータセットのパスとラベルを読み込み
    改ざん画像の枚数に合わせてオリジナル画像をサンプリング
    
    Args:
        config: 設定辞書
    
    Returns:
        image_paths: 画像パスのリスト（バランス調整済み）
        labels: ラベルのリスト（0: オリジナル, 1: 改ざん）
    """
    dataset_config = config['dataset']
    data_path = dataset_config['data_path']
    authentic_folder = dataset_config['authentic_folder']
    tampered_folder = dataset_config['tampered_folder']
    seed = dataset_config.get('seed', 42)
    
    # シード設定（再現性のため）
    random.seed(seed)
    np.random.seed(seed)
    
    image_paths = []
    labels = []
    
    # オリジナル画像読み込み
    authentic_path = os.path.join(data_path, authentic_folder)
    authentic_files = (glob.glob(os.path.join(authentic_path, "*.jpg")) + 
                  glob.glob(os.path.join(authentic_path, "*.tif")))
    
    # 改ざん画像読み込み（全て使用）
    tampered_path = os.path.join(data_path, tampered_folder)
    tampered_files = (glob.glob(os.path.join(tampered_path, "*.jpg")) + 
                  glob.glob(os.path.join(tampered_path, "*.tif")))

    print(f"📁 元データセット情報:")
    print(f"   オリジナル: {len(authentic_files)} 枚")
    print(f"   改ざん: {len(tampered_files)} 枚")
    
    # 改ざん画像の枚数に合わせてオリジナル画像をサンプリング
    target_count = len(tampered_files)
    
    if len(authentic_files) >= target_count:
        # オリジナル画像が十分ある場合：ランダムサンプリング
        sampled_authentic_files = random.sample(authentic_files, target_count)
        print(f"🎯 バランス調整: オリジナル画像を {len(authentic_files)} → {target_count} 枚にサンプリング")
    else:
        # オリジナル画像が不足している場合：エラー
        raise ValueError(
            f"オリジナル画像が不足しています。\n"
            f"必要枚数: {target_count} 枚\n"
            f"利用可能枚数: {len(authentic_files)} 枚\n"
            f"改ざん画像の枚数を減らすか、オリジナル画像を追加してください。"
        )
    
    # データセット構築
    # オリジナル画像（ラベル0）
    image_paths.extend(sampled_authentic_files)
    labels.extend([0] * len(sampled_authentic_files))
    
    # 改ざん画像（ラベル1）- 全て使用
    image_paths.extend(tampered_files)
    labels.extend([1] * len(tampered_files))
    
    print(f"📊 バランス調整後のデータセット:")
    print(f"   オリジナル: {labels.count(0)} 枚")
    print(f"   改ざん: {labels.count(1)} 枚")
    print(f"   合計: {len(image_paths)} 枚")
    print(f"   バランス比: {labels.count(0)}:{labels.count(1)} (1:1)")
    
    # データをシャッフル（バランスを保ったまま）
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)
    
    return list(image_paths), list(labels)


def create_balanced_data_splits(image_paths: List[str], labels: List[int], config: Dict[str, Any]) -> Tuple[
    Tuple[List[str], List[int]], 
    Tuple[List[str], List[int]], 
    Tuple[List[str], List[int]]
]:
    """
    バランス調整済みデータセットを訓練・検証・テストに分割
    分割時もクラスバランスを保持
    
    Args:
        image_paths: 画像パスのリスト
        labels: ラベルのリスト
        config: 設定辞書
    
    Returns:
        (train_paths, train_labels): 訓練データ
        (val_paths, val_labels): 検証データ
        (test_paths, test_labels): テストデータ
    """
    dataset_config = config['dataset']
    train_ratio = dataset_config['train_ratio']
    val_ratio = dataset_config['val_ratio']
    test_ratio = dataset_config['test_ratio']
    seed = dataset_config['seed']
    
    # 比率の確認
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "分割比率の合計が1になりません"
    
    print(f"🔀 バランス保持データ分割開始:")
    print(f"   分割前 - オリジナル: {labels.count(0)}, 改ざん: {labels.count(1)}")
    
    # stratifyを使用してクラスバランスを保持しながら分割
    # まず訓練+検証データとテストデータに分割
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_ratio, 
        random_state=seed, stratify=labels
    )
    
    # 訓練データと検証データに分割
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=adjusted_val_ratio,
        random_state=seed, stratify=train_val_labels
    )
    
    # 分割結果の確認
    print(f"🔀 バランス保持データ分割完了:")
    print(f"   訓練: {len(train_paths)} 枚 (オリジナル: {train_labels.count(0)}, 改ざん: {train_labels.count(1)})")
    print(f"   検証: {len(val_paths)} 枚 (オリジナル: {val_labels.count(0)}, 改ざん: {val_labels.count(1)})")
    print(f"   テスト: {len(test_paths)} 枚 (オリジナル: {test_labels.count(0)}, 改ざん: {test_labels.count(1)})")
    
    # バランス確認
    def check_balance(label_list, name):
        original_count = label_list.count(0)
        tampered_count = label_list.count(1)
        balance_ratio = original_count / tampered_count if tampered_count > 0 else 0
        print(f"   {name}バランス比: {original_count}:{tampered_count} (比率: {balance_ratio:.2f})")
    
    check_balance(train_labels, "訓練")
    check_balance(val_labels, "検証")
    check_balance(test_labels, "テスト")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def create_balanced_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    バランス調整済みデータローダーを作成
    
    Args:
        config: 設定辞書
    
    Returns:
        train_loader: 訓練用データローダー
        val_loader: 検証用データローダー
        test_loader: テスト用データローダー
    """
    # バランス調整済みデータセット読み込み
    image_paths, labels = load_balanced_dataset_paths(config)
    
    # データ分割（バランス保持）
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = create_balanced_data_splits(
        image_paths, labels, config
    )
    
    # データセット作成
    train_dataset = BalancedSaigen90Dataset(train_paths, train_labels, config, mode='train')
    val_dataset = BalancedSaigen90Dataset(val_paths, val_labels, config, mode='val')
    test_dataset = BalancedSaigen90Dataset(test_paths, test_labels, config, mode='test')
    
    # データローダー作成（Docker環境対応）
    training_config = config['training']
    batch_size = training_config['batch_size']
    num_workers = 0  # Docker環境では0に固定
    pin_memory = False  # Docker環境では無効化
    
    # 再現性のためのgenerator設定
    seed = config['dataset']['seed']
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=g  # 再現性のため
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"🚀 バランス調整済みデータローダー作成完了:")
    print(f"   バッチサイズ: {batch_size}")
    print(f"   ワーカー数: {num_workers}")
    print(f"   訓練バッチ数: {len(train_loader)}")
    print(f"   検証バッチ数: {len(val_loader)}")
    print(f"   テストバッチ数: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def analyze_dataset_balance(config: Dict[str, Any]):
    """
    データセットのバランスを分析（実行前の確認用）
    """
    dataset_config = config['dataset']
    data_path = dataset_config['data_path']
    authentic_folder = dataset_config['authentic_folder']
    tampered_folder = dataset_config['tampered_folder']
    
    # オリジナル画像
    authentic_path = os.path.join(data_path, authentic_folder)
    authentic_files = glob.glob(os.path.join(authentic_path, "*.jpg"))
    
    # 改ざん画像
    tampered_path = os.path.join(data_path, tampered_folder)
    tampered_files = glob.glob(os.path.join(tampered_path, "*.jpg"))
    
    print(f"📊 データセットバランス分析:")
    print(f"   オリジナル画像: {len(authentic_files)} 枚")
    print(f"   改ざん画像: {len(tampered_files)} 枚")
    print(f"   元の比率: {len(authentic_files)}:{len(tampered_files)}")
    
    target_count = len(tampered_files)
    print(f"   バランス調整後: 各クラス {target_count} 枚")
    print(f"   総使用枚数: {target_count * 2} 枚")
    
    if len(authentic_files) > len(tampered_files):
        unused_count = len(authentic_files) - len(tampered_files)
        print(f"   ⚠️ オリジナル画像から {unused_count} 枚が使用されません")
    elif len(authentic_files) < len(tampered_files):
        print(f"   ❌ エラー: オリジナル画像が不足しています")
        return False
    else:
        print(f"   ✅ 完全バランス: 調整不要")
    
    return True


if __name__ == "__main__":
    # テスト実行
    print("=" * 60)
    print("バランス調整済みSaigen90 データローダーテスト")
    print("=" * 60)
    
    # 設定読み込み
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # バランス分析
        if not analyze_dataset_balance(config):
            print("❌ データセットのバランス分析でエラーが発生しました。")
            exit(1)
        
        print("\n" + "="*60)
        
        # データローダー作成
        train_loader, val_loader, test_loader = create_balanced_data_loaders(config)
        
        # サンプルデータ確認
        print("\n🔍 サンプルデータ確認:")
        for batch_idx, (features, labels) in enumerate(train_loader):
            print(f"   バッチ {batch_idx + 1}:")
            print(f"     特徴量形状: {features.shape}")  # (B, 23, H, W)
            print(f"     ラベル形状: {labels.shape}")    # (B,)
            print(f"     特徴量範囲: [{features.min():.3f}, {features.max():.3f}]")
            print(f"     ラベル: {labels.tolist()}")
            print(f"     バッチ内バランス: オリジナル {(labels == 0).sum().item()} / 改ざん {(labels == 1).sum().item()}")
            
            if batch_idx >= 2:  # 最初の3バッチのみ確認
                break
        
        print(f"\n✅ バランス調整済みデータローダーテスト成功!")
        print(f"🎯 改ざん画像の枚数に合わせてオリジナル画像をサンプリング")
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()