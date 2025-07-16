#!/usr/bin/env python3
"""
90%精度達成モデル用データローダー
23チャンネル特徴量による画像改ざん検出
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

from utils_90percent import create_23_channel_features, normalize_image


class Saigen90Dataset(Dataset):
    """90%精度再現用データセット（23チャンネル特徴量）"""
    
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
        self.augmentation_config = config.get('augmentation', {})
        
        print(f"📊 Dataset[{mode}]: {len(self.image_paths)} samples")
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
        
        # データ拡張（訓練時のみ）
        if self.mode == 'train':
            image = self._apply_data_augmentation(image)
        
        # 23チャンネル特徴量抽出
        features = create_23_channel_features(image, self.config)
        
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
        # 水平反転
        horizontal_flip = self.augmentation_config.get('horizontal_flip', 0)
        if horizontal_flip > 0 and random.random() < horizontal_flip:
            image = cv2.flip(image, 1)
        
        # 回転
        rotation_range = self.augmentation_config.get('rotation_range', 0)
        if rotation_range > 0:
            angle = random.uniform(-rotation_range, rotation_range)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)
        
        # 明度調整
        brightness_range = self.augmentation_config.get('brightness_range', 0)
        if brightness_range > 0:
            brightness_factor = 1.0 + random.uniform(-brightness_range, brightness_range)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        # コントラスト調整
        contrast_range = self.augmentation_config.get('contrast_range', 0)
        if contrast_range > 0:
            contrast_factor = 1.0 + random.uniform(-contrast_range, contrast_range)
            image = np.clip((image - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        return image


def load_dataset_paths(config: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    """
    データセットのパスとラベルを読み込み
    
    Args:
        config: 設定辞書
    
    Returns:
        image_paths: 画像パスのリスト
        labels: ラベルのリスト（0: オリジナル, 1: 改ざん）
    """
    dataset_config = config['dataset']
    data_path = dataset_config['data_path']
    authentic_folder = dataset_config['authentic_folder']
    tampered_folder = dataset_config['tampered_folder']
    max_samples = dataset_config.get('max_samples_per_class', None)
    
    image_paths = []
    labels = []
    
    # オリジナル画像（ラベル0）
    authentic_path = os.path.join(data_path, authentic_folder)
    authentic_files = glob.glob(os.path.join(authentic_path, "*.jpg"))
    
    if max_samples:
        authentic_files = authentic_files[:max_samples]
    
    image_paths.extend(authentic_files)
    labels.extend([0] * len(authentic_files))
    
    # 改ざん画像（ラベル1）
    tampered_path = os.path.join(data_path, tampered_folder)
    tampered_files = glob.glob(os.path.join(tampered_path, "*.jpg"))
    
    if max_samples:
        tampered_files = tampered_files[:max_samples]
    
    image_paths.extend(tampered_files)
    labels.extend([1] * len(tampered_files))
    
    print(f"📁 データセット読み込み完了:")
    print(f"   オリジナル: {len(authentic_files)} 枚")
    print(f"   改ざん: {len(tampered_files)} 枚")
    print(f"   合計: {len(image_paths)} 枚")
    
    return image_paths, labels


def create_data_splits(image_paths: List[str], labels: List[int], config: Dict[str, Any]) -> Tuple[
    Tuple[List[str], List[int]], 
    Tuple[List[str], List[int]], 
    Tuple[List[str], List[int]]
]:
    """
    データセットを訓練・検証・テストに分割
    
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
    
    # まず訓練データとテストデータに分割
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
    
    print(f"🔀 データ分割完了:")
    print(f"   訓練: {len(train_paths)} 枚 ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"   検証: {len(val_paths)} 枚 ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"   テスト: {len(test_paths)} 枚 ({len(test_paths)/len(image_paths)*100:.1f}%)")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def create_saigen90_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    90%精度再現用データローダーを作成
    
    Args:
        config: 設定辞書
    
    Returns:
        train_loader: 訓練用データローダー
        val_loader: 検証用データローダー
        test_loader: テスト用データローダー
    """
    # データセット読み込み
    image_paths, labels = load_dataset_paths(config)
    
    # データ分割
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = create_data_splits(
        image_paths, labels, config
    )
    
    # データセット作成
    train_dataset = Saigen90Dataset(train_paths, train_labels, config, mode='train')
    val_dataset = Saigen90Dataset(val_paths, val_labels, config, mode='val')
    test_dataset = Saigen90Dataset(test_paths, test_labels, config, mode='test')
    
    # データローダー作成（Docker環境対応）
    training_config = config['training']
    batch_size = training_config['batch_size']
    num_workers = 0  # Docker環境では0に固定
    pin_memory = False  # Docker環境では無効化
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
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
    
    print(f"🚀 データローダー作成完了:")
    print(f"   バッチサイズ: {batch_size}")
    print(f"   ワーカー数: {num_workers}")
    print(f"   訓練バッチ数: {len(train_loader)}")
    print(f"   検証バッチ数: {len(val_loader)}")
    print(f"   テストバッチ数: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # テスト実行
    print("=" * 60)
    print("Saigen90 データローダーテスト")
    print("=" * 60)
    
    # 設定読み込み
    with open('config_90percent.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # データローダー作成
        train_loader, val_loader, test_loader = create_saigen90_data_loaders(config)
        
        # サンプルデータ確認
        print("\n🔍 サンプルデータ確認:")
        for batch_idx, (features, labels) in enumerate(train_loader):
            print(f"   バッチ {batch_idx + 1}:")
            print(f"     特徴量形状: {features.shape}")  # (B, 23, H, W)
            print(f"     ラベル形状: {labels.shape}")    # (B,)
            print(f"     特徴量範囲: [{features.min():.3f}, {features.max():.3f}]")
            print(f"     ラベル: {labels.tolist()}")
            
            if batch_idx >= 2:  # 最初の3バッチのみ確認
                break
        
        print(f"\n✅ データローダーテスト成功!")
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
