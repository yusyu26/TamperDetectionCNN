FROM python:3.8-slim

# バッファリングをオフにして、ログをリアルタイムで表示
ENV PYTHONUNBUFFERED=1

# apt のパッケージリスト更新および最低限のビルドツールをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libglib2.0-0 \
        libgl1       \
        libsm6       \
        libxrender1  \
        libxext6     \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# - numpy, scikit-learn, scikit-image : 基本的な画像処理・機械学習
# - matplotlib, seaborn : 可視化用
# - opencv-python : 画像処理（エッジ検出、色空間変換等）
# - torch, torchvision : PyTorch本体とコンピュータビジョン機能
# - tensorboard : 学習進捗の可視化
# - pyyaml : YAML設定ファイル読み込み
# - tqdm : プログレスバー表示
RUN pip install --no-cache-dir \
        numpy \
        scikit-learn \
        scikit-image \
        matplotlib \
        seaborn \
        opencv-python \
        torch \
        torchvision \
        tensorboard \
        pyyaml \
        tqdm

COPY . /app

CMD ["bash"]
