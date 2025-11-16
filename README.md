# 卒業研究：軽量CNNプログラム使用方法

## 1. 概要

本リポジトリは、卒業研究「軽量なCNNアーキテクチャによる
高精度な画像改ざん検出
」で使用したプログラム一式です。
軽量なCNNモデルを実装し、その学習と評価を行います。

## 2. ファイル構成

本プログラムは以下の構成になっています。
```
├── Dockerfile 　　　　　# Docker環境構築ファイル 
├── requirements.txt   # 必要なPythonライブラリ一覧 
├── README.md          # このファイル 
├── data/ 　　　　　　　 # Kaggleからダウンロード 
│ └── CASIA2/ 
│  ├── Au/ 
│  └── Tp/ 
└── 軽量CNN/                    # ソースコード 
  ├── train.py                 # 学習スクリプト 
  ├── evaluate.py              # 評価スクリプト 
  ├── model.py                 # モデル（CNNアーキテクチャ）定義 
  ├── balanced_data_loader.py  # データローダー 
  ├── utils_90percent.py       # ユーティリティ・ヘルパー関数 
  └── config.yaml              # 学習/評価の設定ファイル
```

## 3. 環境構築

今回は全ての環境で同じように動作することを保証するために、実行環境にDockerを使用します。

### Dockerを使用した環境構築し手順

1.  **Dockerイメージのビルド:**
    
    ターミナルでこの `Dockerfile` があるディレクトリに移動し、以下のコマンドを実行します。
    
    ```bash
    docker build -t keiryo-cnn .
    ```

2.  **Dockerコンテナの実行:**
    
    学習データの配置（後述の「4. データセットの準備」参照）が完了したら、以下のコマンドでコンテナを起動します。
    
    ```bash
    # コンテナの実行
    docker run -d --name cnn-container
    ```
    
    コンテナが起動したら、`# 5. 実行方法` に進んでください。


## 4. データセットの準備

このプログラムを実行するには、学習用および評価用のデータセットが所定の場所に配置されている必要があります。

1.  プロジェクトのルート（`Dockerfile` と同じ階層）に `data` フォルダを**手動で作成**してください。
2.  `data` フォルダ内に、`config.yaml` の `data_dir` で指定された構造に従ってデータを配置します。

    （例）
    
    ```
    data/CASIA2
    ├── Au/
    │   ├── image001.png
    │   └──  ...
    └── Tp/
        ├── image001.png
        └──  ...

    ```
    

## 5. 実行方法

環境構築とデータセットの準備が完了したら、以下のスクリプトを実行します。


### 学習 (Training)

以下のコマンドで学習を開始します。
学習済みモデルは `models/` フォルダ（自動生成）に保存されます。

```bash
# コンテナに接続
docker exec -it cnn-container bash
# 学習スクリプトの実行
python train.py --config config.yaml
```
### 評価 (Evaluation)

```bash
# コンテナに接続
docker exec -it cnn-container bash
# 評価用スクリプトの実行
python evaluate.py --config config.yaml
```
