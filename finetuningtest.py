import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
from tensorflow.keras.applications import RegNetX002, RegNetX004, RegNetX006
from tensorflow.keras.applications import ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge

# モデルの保存用フォルダの指定
MODEL_SAVE_DIR = "model"

# モデルの設定
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

# データセットの設定
DATASET_DIR = "ReferencePicture/9panel3"

# 画像の読み込み関数
def load_images(dataset_dir):
    images = []
    for file in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, file)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = np.array(image)
        images.append(image)
    return images

# モデルの学習関数
def train_model(model, images):
    # データセットを分割
    (x_train, y_train), (x_test, y_test) = train_test_split(images, train_size=0.8)

    # 損失関数と最適化アルゴリズムの設定
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # モデルの学習
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # モデルの評価
    model.evaluate(x_test, y_test)

    # モデルの保存
    model.save(os.path.join(MODEL_SAVE_DIR, model.name))

# アプリケーションのメインループ
if __name__ == "__main__":
    # アプリケーションのインスタンス化
    app = ImageRecognitionApp()

    # モデルの読み込み
    model_name = app.model_name
    model = MODEL_CLASSES[model_name]()

    # 画像の読み込み
    images = load_images(DATASET_DIR)

    # モデルの学習
    train_model(model, images)

    # アプリケーションの実行
    app.mainloop()
