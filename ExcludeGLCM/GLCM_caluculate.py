import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict, Any
import os
from utils_90percent import (
    convert_color_space,
    compute_glcm_features,
    apply_scharr_edge_detection,
    compute_residual_features
)

def analyze_single_image_glcm(image_path: str, config: Dict[str, Any]):
    """
    指定した画像のGLCM特徴量を詳細に分析・可視化
    
    Args:
        image_path: 分析する画像のパス
        config: 設定辞書
    """
    print(f"🔍 Analyzing GLCM for image: {os.path.basename(image_path)}")
    print("=" * 60)
    
    # 1. 画像を読み込み
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    print(f"📊 Image loaded: {image.shape}")
    
    # 2. 前処理（色空間変換）
    preprocessing_config = config['preprocessing']
    color_space = preprocessing_config['color_space']
    channels = preprocessing_config['channels']['use_channels']
    
    converted_image = convert_color_space(image, color_space, channels)
    cr_channel = converted_image[:, :, 0]
    cb_channel = converted_image[:, :, 1]
    
    print(f"📊 Converted to {color_space.upper()}: Cr={cr_channel.shape}, Cb={cb_channel.shape}")
    
    # 3. GLCM特徴量を計算
    glcm_config = preprocessing_config['glcm']
    
    print("\n🧮 Computing GLCM features...")
    glcm_cr = compute_glcm_features(
        cr_channel,
        distances=glcm_config['distances'],
        angles=glcm_config['angles'],
        levels=glcm_config['levels'],
        properties=glcm_config['properties'][:2]
    )
    
    glcm_cb = compute_glcm_features(
        cb_channel,
        distances=glcm_config['distances'],
        angles=glcm_config['angles'],
        levels=glcm_config['levels'],
        properties=glcm_config['properties'][:2]
    )
    
    # 4. 詳細分析
    analyze_glcm_channels(cr_channel, glcm_cr, "Cr", glcm_config)
    analyze_glcm_channels(cb_channel, glcm_cb, "Cb", glcm_config)
    
    # 5. 可視化
    visualize_glcm_analysis(image, cr_channel, cb_channel, glcm_cr, glcm_cb, image_path)
    
    return glcm_cr, glcm_cb

def analyze_glcm_channels(original_channel: np.ndarray, 
                         glcm_features: np.ndarray, 
                         channel_name: str,
                         glcm_config: Dict[str, Any]):
    """
    GLCMチャンネルの詳細分析
    """
    print(f"\n📈 {channel_name} Channel GLCM Analysis:")
    print("-" * 40)
    
    distances = glcm_config['distances']
    angles = glcm_config['angles']
    properties = glcm_config['properties'][:2]
    
    total_channels = len(distances) * len(angles) * len(properties)
    same_value_count = 0
    
    channel_idx = 0
    for prop_idx, prop in enumerate(properties):
        print(f"\n  🔸 {prop.upper()} features:")
        for d_idx, distance in enumerate(distances):
            for a_idx, angle in enumerate(angles):
                channel = glcm_features[:, :, channel_idx]
                unique_values = np.unique(channel)
                
                print(f"    Distance={distance}, Angle={angle}°:")
                print(f"      Unique values: {len(unique_values)}")
                print(f"      Value range: [{channel.min():.6f}, {channel.max():.6f}]")
                print(f"      Standard deviation: {channel.std():.6f}")
                
                # サンプル値を表示
                h, w = channel.shape
                sample_positions = [
                    (0, 0), (h//4, w//4), (h//2, w//2), (3*h//4, 3*w//4), (h-1, w-1)
                ]
                sample_values = [channel[pos] for pos in sample_positions]
                print(f"      Sample values: {[f'{v:.3f}' for v in sample_values]}")
                
                if len(unique_values) == 1:
                    print(f"      ❌ ALL PIXELS SAME VALUE: {unique_values[0]:.6f}")
                    same_value_count += 1
                else:
                    print(f"      ✅ Has spatial variation")
                
                channel_idx += 1
    
    print(f"\n📊 {channel_name} Summary:")
    print(f"   Total channels: {total_channels}")
    print(f"   Constant channels: {same_value_count}")
    print(f"   Variable channels: {total_channels - same_value_count}")
    
    if same_value_count == total_channels:
        print(f"   ❌ ALL {channel_name} GLCM channels are CONSTANT!")
    elif same_value_count > 0:
        print(f"   ⚠️  {same_value_count}/{total_channels} {channel_name} channels are constant")
    else:
        print(f"   ✅ All {channel_name} channels have spatial variation")

def visualize_glcm_analysis(original_image: np.ndarray,
                           cr_channel: np.ndarray,
                           cb_channel: np.ndarray,
                           glcm_cr: np.ndarray,
                           glcm_cb: np.ndarray,
                           image_path: str):
    """
    GLCM分析結果の可視化
    """
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'GLCM Analysis: {os.path.basename(image_path)}', fontsize=16, fontweight='bold')
    
    # レイアウト: 4行6列
    
    # 1行目: 元画像と色空間変換結果
    ax1 = plt.subplot(4, 6, 1)
    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2 = plt.subplot(4, 6, 2)
    ax2.imshow(cr_channel, cmap='gray')
    ax2.set_title('Cr Channel')
    ax2.axis('off')
    
    ax3 = plt.subplot(4, 6, 3)
    ax3.imshow(cb_channel, cmap='gray')
    ax3.set_title('Cb Channel')
    ax3.axis('off')
    
    # Crチャンネルのヒストグラム
    ax4 = plt.subplot(4, 6, 4)
    ax4.hist(cr_channel.flatten(), bins=50, alpha=0.7, color='red')
    ax4.set_title('Cr Histogram')
    ax4.set_xlabel('Pixel Value')
    ax4.set_ylabel('Frequency')
    
    # Cbチャンネルのヒストグラム
    ax5 = plt.subplot(4, 6, 5)
    ax5.hist(cb_channel.flatten(), bins=50, alpha=0.7, color='blue')
    ax5.set_title('Cb Histogram')
    ax5.set_xlabel('Pixel Value')
    ax5.set_ylabel('Frequency')
    
    # 空白
    plt.subplot(4, 6, 6)
    plt.axis('off')
    
    # 2-3行目: Cr GLCM特徴量（最初の8チャンネル）
    for i in range(8):
        row = 2 + i // 6
        col = (i % 6) + 1
        ax = plt.subplot(4, 6, (row-1)*6 + col)
        
        channel = glcm_cr[:, :, i]
        im = ax.imshow(channel, cmap='viridis')
        ax.set_title(f'Cr GLCM Ch{i+1}\nVal: {channel[0,0]:.3f}')
        ax.axis('off')
        
        # カラーバーを追加
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 統計情報をテキストで追加
        unique_count = len(np.unique(channel))
        if unique_count == 1:
            ax.text(0.5, 0.95, 'CONSTANT', transform=ax.transAxes, 
                   ha='center', va='top', color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4行目: Cb GLCM特徴量（最初の6チャンネル）
    for i in range(6):
        ax = plt.subplot(4, 6, 19 + i)
        
        channel = glcm_cb[:, :, i]
        im = ax.imshow(channel, cmap='plasma')
        ax.set_title(f'Cb GLCM Ch{i+1}\nVal: {channel[0,0]:.3f}')
        ax.axis('off')
        
        # カラーバーを追加
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 統計情報をテキストで追加
        unique_count = len(np.unique(channel))
        if unique_count == 1:
            ax.text(0.5, 0.95, 'CONSTANT', transform=ax.transAxes, 
                   ha='center', va='top', color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 統計サマリーも別途表示
    create_glcm_summary_plot(glcm_cr, glcm_cb, image_path)

def create_glcm_summary_plot(glcm_cr: np.ndarray, glcm_cb: np.ndarray, image_path: str):
    """
    GLCM統計サマリーのプロット
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'GLCM Statistics Summary: {os.path.basename(image_path)}', fontsize=14)
    
    # Cr GLCM チャンネルごとの統計
    cr_stats = []
    for i in range(min(8, glcm_cr.shape[2])):
        channel = glcm_cr[:, :, i]
        stats = {
            'channel': i+1,
            'unique_values': len(np.unique(channel)),
            'std': channel.std(),
            'min': channel.min(),
            'max': channel.max(),
            'mean': channel.mean()
        }
        cr_stats.append(stats)
    
    # Cb GLCM チャンネルごとの統計
    cb_stats = []
    for i in range(min(8, glcm_cb.shape[2])):
        channel = glcm_cb[:, :, i]
        stats = {
            'channel': i+1,
            'unique_values': len(np.unique(channel)),
            'std': channel.std(),
            'min': channel.min(),
            'max': channel.max(),
            'mean': channel.mean()
        }
        cb_stats.append(stats)
    
    # プロット1: ユニーク値の数
    channels = [s['channel'] for s in cr_stats]
    unique_vals_cr = [s['unique_values'] for s in cr_stats]
    unique_vals_cb = [s['unique_values'] for s in cb_stats]
    
    x = np.arange(len(channels))
    width = 0.35
    
    ax1.bar(x - width/2, unique_vals_cr, width, label='Cr', alpha=0.8, color='red')
    ax1.bar(x + width/2, unique_vals_cb, width, label='Cb', alpha=0.8, color='blue')
    ax1.set_xlabel('GLCM Channel')
    ax1.set_ylabel('Number of Unique Values')
    ax1.set_title('Unique Values per Channel')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Ch{i}' for i in channels])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # プロット2: 標準偏差
    std_cr = [s['std'] for s in cr_stats]
    std_cb = [s['std'] for s in cb_stats]
    
    ax2.bar(x - width/2, std_cr, width, label='Cr', alpha=0.8, color='red')
    ax2.bar(x + width/2, std_cb, width, label='Cb', alpha=0.8, color='blue')
    ax2.set_xlabel('GLCM Channel')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Standard Deviation per Channel')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Ch{i}' for i in channels])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # プロット3: 値の範囲
    range_cr = [s['max'] - s['min'] for s in cr_stats]
    range_cb = [s['max'] - s['min'] for s in cb_stats]
    
    ax3.bar(x - width/2, range_cr, width, label='Cr', alpha=0.8, color='red')
    ax3.bar(x + width/2, range_cb, width, label='Cb', alpha=0.8, color='blue')
    ax3.set_xlabel('GLCM Channel')
    ax3.set_ylabel('Value Range (Max - Min)')
    ax3.set_title('Value Range per Channel')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Ch{i}' for i in channels])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # プロット4: 平均値
    mean_cr = [s['mean'] for s in cr_stats]
    mean_cb = [s['mean'] for s in cb_stats]
    
    ax4.bar(x - width/2, mean_cr, width, label='Cr', alpha=0.8, color='red')
    ax4.bar(x + width/2, mean_cb, width, label='Cb', alpha=0.8, color='blue')
    ax4.set_xlabel('GLCM Channel')
    ax4.set_ylabel('Mean Value')
    ax4.set_title('Mean Value per Channel')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Ch{i}' for i in channels])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# メイン実行部分
if __name__ == "__main__":
    # 設定
    config = {
        'preprocessing': {
            'color_space': 'ycrcb',
            'channels': {'use_channels': ['cr', 'cb']},
            'glcm': {
                'distances': [1, 2],
                'angles': [0, 45, 90, 135],
                'levels': 16,
                'properties': ['contrast', 'homogeneity']
            }
        }
    }
    
    # 画像パスを指定（実際のパスに変更してください）
    image_path = "../data/others/IMG_9167.jpg"  # ← ここを実際の画像パスに変更
    
    # 分析実行
    try:
        glcm_cr, glcm_cb = analyze_single_image_glcm(image_path, config)
        print("\n✅ Analysis completed successfully!")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")