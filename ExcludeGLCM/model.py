import torch
import torch.nn as nn

class SaigenCNN(nn.Module):
    def __init__(self, in_channels=23, num_classes=2):
        super(SaigenCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1: 23チャンネル -> 32チャンネル, 画像サイズ 128x128 -> 64x64
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32チャンネル -> 64チャンネル, 画像サイズ 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64チャンネル -> 128チャンネル, 画像サイズ 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 128チャンネル -> 256チャンネル, 画像サイズ 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 最終的な特徴マップのサイズは 256 * 8 * 8 = 16384
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 畳み込み層を通過させる
        x = self.conv_layers(x)
        
        # 全結合層への入力のために、テンソルを1次元に平坦化する
        x = torch.flatten(x, 1)
        
        # 全結合層を通過させる
        x = self.fc_layers(x)
        
        return x

if __name__ == '__main__':
    # モデルが正しく定義できているか、簡単なテスト
    # 23チャンネル, 128x128のダミーデータを作成
    dummy_input = torch.randn(1, 23, 128, 128) # (バッチサイズ, チャンネル, 高さ, 幅)
    model = SaigenCNN()
    output = model(dummy_input)
    
    print("モデルの構造:")
    print(model)
    print("\nダミー入力の形状:", dummy_input.shape)
    print("モデルからの出力の形状:", output.shape)
    # 期待される出力形状: (1, 2) -> 正しければOK
    assert output.shape == (1, 2)
    print("\nモデルの構造チェックOK！")