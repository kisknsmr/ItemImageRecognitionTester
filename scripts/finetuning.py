"""
ファインチューニングスクリプト
独自データでのモデル追加学習
"""
import os
import argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import (
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2,
    RegNetX002, RegNetX004, RegNetX006,
    ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
)


# 利用可能なモデルクラス
MODEL_CLASSES = {
    "EfficientNetV2B0": EfficientNetV2B0,
    "EfficientNetV2B1": EfficientNetV2B1,
    "EfficientNetV2B2": EfficientNetV2B2,
    "RegNetX002": RegNetX002,
    "RegNetX004": RegNetX004,
    "RegNetX006": RegNetX006,
    "ConvNeXtSmall": ConvNeXtSmall,
    "ConvNeXtBase": ConvNeXtBase,
    "ConvNeXtLarge": ConvNeXtLarge,
    "ConvNeXtXLarge": ConvNeXtXLarge
}


def load_images(dataset_dir: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    ディレクトリから画像を読み込み
    
    Args:
        dataset_dir: 画像ディレクトリ
        target_size: リサイズサイズ
        
    Returns:
        画像のnumpy配列
    """
    images = []
    for file in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, file)
        if not os.path.isfile(image_path):
            continue
        try:
            image = Image.open(image_path)
            image = image.resize(target_size)
            image = np.array(image)
            images.append(image)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    return np.array(images)


def train_model(model, images: np.ndarray, model_save_dir: str = "model",
                epochs: int = 10, batch_size: int = 32, 
                learning_rate: float = 0.001):
    """
    モデルを学習
    
    Args:
        model: Kerasモデルインスタンス
        images: 学習画像配列
        model_save_dir: モデル保存ディレクトリ
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
    """
    # データセットを分割
    x_train, x_test = train_test_split(images, train_size=0.8, random_state=42)
    
    # ラベル生成（ダミー - 実際は適切なラベルが必要）
    num_classes = 10  # 仮のクラス数
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(0, num_classes, len(x_train)), num_classes
    )
    y_test = tf.keras.utils.to_categorical(
        np.random.randint(0, num_classes, len(x_test)), num_classes
    )
    
    # 損失関数と最適化アルゴリズムの設定
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # モデルの学習
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
              validation_data=(x_test, y_test))
    
    # モデルの評価
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # モデルの保存
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, model.name)
    model.save(save_path)
    print(f"Model saved to: {save_path}")


def main():
    """コマンドラインエントリーポイント"""
    parser = argparse.ArgumentParser(description='モデルのファインチューニング')
    parser.add_argument('--model', '-m', default='EfficientNetV2B0',
                        choices=list(MODEL_CLASSES.keys()),
                        help='使用するモデル')
    parser.add_argument('--dataset', '-d', default='ReferencePicture/9panel3',
                        help='学習データディレクトリ')
    parser.add_argument('--output', '-o', default='model',
                        help='モデル保存ディレクトリ')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='エポック数')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='バッチサイズ')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001,
                        help='学習率')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    ModelClass = MODEL_CLASSES[args.model]
    model = ModelClass(weights='imagenet')
    
    print(f"Loading images from: {args.dataset}")
    images = load_images(args.dataset)
    
    if len(images) == 0:
        print("Error: No images found in dataset directory")
        return 1
    
    print(f"Loaded {len(images)} images")
    print(f"Starting training for {args.epochs} epochs...")
    
    train_model(
        model, images, 
        model_save_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
