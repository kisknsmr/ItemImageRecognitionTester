### 1. README.md

```markdown
# Deep Learning Image Recognition & Analysis Suite

このプロジェクトは、TensorFlow/Kerasを利用して、複数の深層学習モデルによる画像認識精度の比較、学習用データの作成（画像分割）、およびモデルのファインチューニングを行うためのツール群です。

## 構成ファイル

1. **main.py**: 
   - 20種類以上の学習済みモデルを切り替え可能なGUIアプリケーション。
   - 参照画像（ターゲット）と9枚の候補画像を比較し、セマンティックマッチング（意味的合致）による上位3位をハイライト表示します。
2. **Split9panels.py**: 
   - 1枚の画像を3x3の9パネルに自動分割し、日時ベースのファイル名で保存するデータ作成用スクリプト。
3. **finetuningtest.py**: 
   - EfficientNetV2やConvNeXtなどの最新モデルに対して、独自の画像データで追加学習（ファインチューニング）を試行するためのテストスクリプト。

## セットアップ

### 必要条件
- Python 3.9以上推奨
- NVIDIA GPU (CUDA/cuDNN環境) 推奨 ※CPUでも動作しますが、推論・学習に時間を要します。

### インストール
```bash
pip install -r requirements.txt

```

## 使い方

### 画像認識アプリの起動

```bash
python main.py

```

* 左上の画像が「参照」となり、下の9枚から類似したものをAIが判定します。
* 右側のパネルからモデル名をクリックすると、非同期でモデルがロードされ、即座に再計算が始まります。

### 画像の分割

`sample.png` を用意し、以下のコマンドを実行すると `learningdata` フォルダに分割画像が生成されます。

```bash
python Split9panels.py

```

## 注意事項

* `main.py` 実行には、あらかじめ `ReferencePicture/9panel3` フォルダ内に `0.png` 〜 `9.png` の画像が配置されている必要があります。

```

---

### 2. requirements.txt

```text
# Core Machine Learning
tensorflow>=2.10.0
numpy
scikit-learn

# Image Processing
Pillow

# GUI Support
# (tkinterは標準ライブラリですが、画像表示にPillowが必要です)

```

---

### 🛠 ログ出力に関する適用（ユーザー設定）

ご指定いただいた「プログラミング生成時のログ出力（イミディエイトウィンドウ）」仕様に基づき、各ファイルの現状を確認しました。

* **`main.py`**: 現在、`logging.info` と `logging.error` が実装されています。モデルのロード完了時や予測結果の確定時に、IDEのコンソールへ詳細が出力されるようになっています。
* **`finetuningtest.py`**: 非常にシンプルな構成ですが、学習の進捗（LossやAccuracy）は標準出力に流れるよう設計されています。