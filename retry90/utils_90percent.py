#!/usr/bin/env python3
"""
90%精度達成モデル用ユーティリティ関数
23チャンネル特徴量抽出のための関数群
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def convert_color_space(image: np.ndarray, color_space: str, channels: List[str]) -> np.ndarray:
    """
    色空間変換とチャンネル抽出
    
    Args:
        image: RGB画像 (H, W, 3)
        color_space: 色空間 ("ycrcb", "lab", "hsv", etc.)
        channels: 使用するチャンネル ["cr", "cb", "y", etc.]
    
    Returns:
        変換後の画像 (H, W, len(channels))
    """
    if color_space.lower() == "ycrcb":
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        channel_map = {"y": 0, "cr": 1, "cb": 2}
    elif color_space.lower() == "lab":
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        channel_map = {"l": 0, "a": 1, "b": 2}
    elif color_space.lower() == "hsv":
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        channel_map = {"h": 0, "s": 1, "v": 2}
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    
    # 指定されたチャンネルを抽出
    selected_channels = []
    for channel in channels:
        if channel.lower() in channel_map:
            idx = channel_map[channel.lower()]
            selected_channels.append(converted[:, :, idx])
        else:
            raise ValueError(f"Invalid channel '{channel}' for color space '{color_space}'")
    
    # チャンネル次元を追加
    if len(selected_channels) == 1:
        return selected_channels[0][:, :, np.newaxis]
    else:
        return np.stack(selected_channels, axis=2)


def apply_scharr_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Scharrフィルタによるエッジ検出
    
    Args:
        image: グレースケール画像 (H, W)
    
    Returns:
        エッジ画像 (H, W)
    """
    # Scharrフィルタを適用
    grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)  # X方向勾配
    grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)  # Y方向勾配
    
    # 勾配の大きさを計算
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 0-255に正規化
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude


def compute_glcm_features(image: np.ndarray, 
                         distances: List[int] = [1, 2],
                         angles: List[int] = [0, 45, 90, 135],
                         levels: int = 16,
                         properties: List[str] = ['contrast', 'homogeneity']) -> np.ndarray:
    """
    GLCM特徴量の計算
    
    Args:
        image: グレースケール画像 (H, W)
        distances: 距離のリスト
        angles: 角度のリスト（度）
        levels: 量子化レベル
        properties: 特徴量のリスト
    
    Returns:
        GLCM特徴量マップ (H, W, len(distances) * len(angles) * len(properties))
    """
    # 入力画像を量子化
    if image.max() > levels - 1:
        quantized = (image / (256 / levels)).astype(np.uint8)
        quantized = np.clip(quantized, 0, levels - 1)
    else:
        quantized = image.astype(np.uint8)
    
    # 角度をラジアンに変換
    angles_rad = [np.deg2rad(angle) for angle in angles]
    
    # GLCMを計算
    glcm = graycomatrix(quantized, 
                       distances=distances, 
                       angles=angles_rad, 
                       levels=levels,
                       symmetric=True, 
                       normed=True)
    
    feature_maps = []
    
    # 各特徴量を計算
    for prop in properties:
        feature_map = graycoprops(glcm, prop)
        
        # 各距離・角度の組み合わせに対して特徴量マップを作成
        for d_idx in range(len(distances)):
            for a_idx in range(len(angles)):
                # 特徴量値をブロードキャストして画像サイズに拡張
                feature_value = feature_map[d_idx, a_idx]
                feature_channel = np.full(image.shape, feature_value, dtype=np.float32)
                feature_maps.append(feature_channel)
    
    # すべての特徴量マップを結合
    glcm_features = np.stack(feature_maps, axis=2)
    
    # 正規化 (0-255)
    glcm_features = ((glcm_features - glcm_features.min()) / 
                    (glcm_features.max() - glcm_features.min() + 1e-8) * 255)
    
    return glcm_features.astype(np.uint8)


def compute_residual_features(image: np.ndarray, methods: List[str] = ["median", "gaussian"]) -> np.ndarray:
    """
    残差特徴量の計算
    
    Args:
        image: グレースケール画像 (H, W)
        methods: 使用する手法のリスト
    
    Returns:
        残差特徴量 (H, W, len(methods))
    """
    residuals = []
    
    for method in methods:
        if method == "median":
            # メディアンフィルタによる残差
            filtered = cv2.medianBlur(image, 5)
            residual = image.astype(np.float32) - filtered.astype(np.float32)
        elif method == "gaussian":
            # ガウシアンフィルタによる残差
            filtered = cv2.GaussianBlur(image, (5, 5), 1.0)
            residual = image.astype(np.float32) - filtered.astype(np.float32)
        else:
            raise ValueError(f"Unknown residual method: {method}")
        
        # 残差を0-255範囲に正規化
        residual = residual + 128  # 負の値を正に移動
        residual = np.clip(residual, 0, 255).astype(np.uint8)
        residuals.append(residual)
    
    return np.stack(residuals, axis=2)


def create_23_channel_features(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    90%精度モデルの23チャンネル特徴量を作成
    
    構成:
    - 基本色差チャンネル: 2ch (Cr, Cb)
    - エッジ検出: 1ch (CrチャンネルのScharr)
    - GLCM特徴量: 16ch (Cr, Cb各8ch)
    - 残差特徴量: 4ch (Cr, Cb各2ch)
    合計: 23チャンネル
    
    Args:
        image: RGB画像 (H, W, 3)
        config: 設定辞書
    
    Returns:
        23チャンネル特徴量 (H, W, 23)
    """
    preprocessing_config = config['preprocessing']
    
    # 1. 色空間変換 (2ch: Cr, Cb)
    color_space = preprocessing_config['color_space']
    channels = preprocessing_config['channels']['use_channels']
    converted_image = convert_color_space(image, color_space, channels)
    
    cr_channel = converted_image[:, :, 0]  # Crチャンネル
    cb_channel = converted_image[:, :, 1]  # Cbチャンネル
    
    feature_channels = []
    feature_channels.append(cr_channel[:, :, np.newaxis])    # ch 0: Cr
    feature_channels.append(cb_channel[:, :, np.newaxis])    # ch 1: Cb
    
    # 2. エッジ検出：Crチャンネルに適用 (1ch)
    edge_image = apply_scharr_edge_detection(cr_channel)
    feature_channels.append(edge_image[:, :, np.newaxis])     # ch 2: Crエッジ
    
    # 3. GLCM特徴量：Cr、Cbそれぞれに適用 (16ch)
    glcm_config = preprocessing_config['glcm']
    
    # Crチャンネル用GLCM (8ch)
    glcm_cr = compute_glcm_features(
        cr_channel,
        distances=glcm_config['distances'],
        angles=glcm_config['angles'],
        levels=glcm_config['levels'],
        properties=glcm_config['properties'][:2]  # 2特徴量のみ使用して8chに調整
    )[:, :, :8]  # 最初の8チャンネルのみ
    
    # Cbチャンネル用GLCM (8ch)
    glcm_cb = compute_glcm_features(
        cb_channel,
        distances=glcm_config['distances'],
        angles=glcm_config['angles'],
        levels=glcm_config['levels'],
        properties=glcm_config['properties'][:2]  # 2特徴量のみ使用して8chに調整
    )[:, :, :8]  # 最初の8チャンネルのみ
    
    # GLCM特徴量を個別チャンネルとして追加
    for i in range(8):
        feature_channels.append(glcm_cr[:, :, i:i+1])         # ch 3-10: Cr GLCM
    for i in range(8):
        feature_channels.append(glcm_cb[:, :, i:i+1])         # ch 11-18: Cb GLCM
    
    # 4. 残差特徴量：Cr、Cbそれぞれに適用 (4ch)
    residual_config = preprocessing_config['residual_features']
    methods = residual_config['methods']
    
    # Crチャンネル残差 (2ch)
    residual_cr = compute_residual_features(cr_channel, methods)
    feature_channels.append(residual_cr[:, :, 0:1])          # ch 19: Cr median残差
    feature_channels.append(residual_cr[:, :, 1:2])          # ch 20: Cr gaussian残差
    
    # Cbチャンネル残差 (2ch)
    residual_cb = compute_residual_features(cb_channel, methods)
    feature_channels.append(residual_cb[:, :, 0:1])          # ch 21: Cb median残差
    feature_channels.append(residual_cb[:, :, 1:2])          # ch 22: Cb gaussian残差
    
    # 全チャンネルを結合 (H, W, 23)
    multi_channel_image = np.concatenate(feature_channels, axis=2)
    
    # チャンネル数の確認
    assert multi_channel_image.shape[2] == 23, f"Expected 23 channels, got {multi_channel_image.shape[2]}"
    
    return multi_channel_image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    画像の正規化
    
    Args:
        image: 入力画像 (H, W, C)
    
    Returns:
        正規化済み画像 (H, W, C)
    """
    # 0-1範囲に正規化
    normalized = image.astype(np.float32) / 255.0
    return normalized


def calculate_accuracy(outputs: np.ndarray, targets: np.ndarray) -> float:
    """
    精度計算
    
    Args:
        outputs: 予測結果 (N, num_classes)
        targets: 正解ラベル (N,)
    
    Returns:
        精度
    """
    predictions = np.argmax(outputs, axis=1)
    accuracy = np.mean(predictions == targets)
    return accuracy


if __name__ == "__main__":
    # テスト用
    import yaml
    
    # サンプル画像（128x128のRGB）
    test_image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    
    # サンプル設定
    test_config = {
        'preprocessing': {
            'color_space': 'ycrcb',
            'channels': {'use_channels': ['cr', 'cb']},
            'edge_detection': {'method': 'scharr', 'apply_to_channels': 'cr'},
            'glcm': {
                'apply_to_channels': 'both',
                'distances': [1, 2],
                'angles': [0, 45, 90, 135],
                'levels': 16,
                'properties': ['contrast', 'homogeneity']
            },
            'residual_features': {
                'methods': ['median', 'gaussian'],
                'apply_to_channels': 'both'
            }
        }
    }
    
    # 23チャンネル特徴量の作成をテスト
    features = create_23_channel_features(test_image, test_config)
    print(f"✅ 23チャンネル特徴量作成成功: {features.shape}")
    print(f"   期待値: (128, 128, 23)")
    print(f"   実際値: {features.shape}")
    
    # 正規化テスト
    normalized = normalize_image(features)
    print(f"✅ 正規化成功: {normalized.shape}, 範囲: [{normalized.min():.3f}, {normalized.max():.3f}]")
