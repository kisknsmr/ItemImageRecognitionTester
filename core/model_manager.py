"""
モデル管理モジュール
TensorFlow/Kerasモデルの設定・ロード・キャッシュを管理する
"""
import logging
from concurrent.futures import ThreadPoolExecutor
import os

# TensorFlowモデルのインポート
from tensorflow.keras.applications import (
    ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2,
    VGG16, VGG19, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, Xception,
    DenseNet121, DenseNet169, DenseNet201, NASNetMobile, NASNetLarge,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4,
    EfficientNetB5, EfficientNetB6, EfficientNetB7,
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2,
)
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet_v2
from tensorflow.keras.applications import (
    vgg16, vgg19, inception_v3, inception_resnet_v2, mobilenet, mobilenet_v2, xception,
    densenet, nasnet, efficientnet,
)


class ModelManager:
    """TensorFlow/Kerasモデルの管理クラス"""
    
    # 入力サイズマッピング
    SIZE_MAP = {
        "224": (224, 224), "299": (299, 299), "331": (331, 331), "600": (600, 600),
        "240": (240, 240), "260": (260, 260), "300": (300, 300), "380": (380, 380),
        "456": (456, 456), "528": (528, 528),
    }
    
    # 利用可能な全モデルクラス
    ALL_MODEL_CLASSES = {
        "ResNet50": ResNet50, "ResNet101": ResNet101, "ResNet152": ResNet152,
        "ResNet50V2": ResNet50V2, "ResNet101V2": ResNet101V2, "ResNet152V2": ResNet152V2,
        "VGG16": VGG16, "VGG19": VGG19, "InceptionV3": InceptionV3, "InceptionResNetV2": InceptionResNetV2,
        "MobileNet": MobileNet, "MobileNetV2": MobileNetV2, "Xception": Xception,
        "DenseNet121": DenseNet121, "DenseNet169": DenseNet169, "DenseNet201": DenseNet201,
        "NASNetMobile": NASNetMobile, "NASNetLarge": NASNetLarge,
        "EfficientNetB0": EfficientNetB0, "EfficientNetB1": EfficientNetB1, "EfficientNetB2": EfficientNetB2,
        "EfficientNetB3": EfficientNetB3, "EfficientNetB4": EfficientNetB4, "EfficientNetB5": EfficientNetB5,
        "EfficientNetB6": EfficientNetB6, "EfficientNetB7": EfficientNetB7,
        "EfficientNetV2B0": EfficientNetV2B0, "EfficientNetV2B1": EfficientNetV2B1,
        "EfficientNetV2B2": EfficientNetV2B2,
    }
    
    # モデル設定 (前処理関数, 入力サイズ, デコード関数)
    MODEL_CONFIG = {
        "ResNet50": (preprocess_resnet50, SIZE_MAP["224"], decode_predictions),
        "VGG16": (vgg16.preprocess_input, SIZE_MAP["224"], decode_predictions),
        "MobileNetV2": (mobilenet_v2.preprocess_input, SIZE_MAP["224"], decode_predictions),
        "EfficientNetB0": (efficientnet.preprocess_input, SIZE_MAP["224"], decode_predictions),
        "Xception": (xception.preprocess_input, SIZE_MAP["299"], decode_predictions),
        "ResNet101": (preprocess_resnet, SIZE_MAP["224"], decode_predictions),
        "EfficientNetV2B1": (efficientnet.preprocess_input, SIZE_MAP["240"], decode_predictions),
        "NASNetLarge": (nasnet.preprocess_input, SIZE_MAP["331"], decode_predictions),
        "DenseNet121": (densenet.preprocess_input, SIZE_MAP["224"], decode_predictions),
    }
    
    # 起動時にロードするモデル
    INITIAL_MODELS = ["ResNet50", "VGG16", "MobileNetV2", "EfficientNetB0", "Xception"]
    
    def __init__(self):
        self.loaded_models = {}
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
    
    def get_model_config(self, model_name: str):
        """指定モデルの設定を取得"""
        return self.MODEL_CONFIG.get(model_name)
    
    def get_all_model_names(self) -> list:
        """全モデル名のリストを取得"""
        return sorted(list(self.ALL_MODEL_CLASSES.keys()))
    
    def get_loaded_model_names(self) -> list:
        """ロード済みモデル名のリストを取得"""
        return sorted(list(self.loaded_models.keys()))
    
    def get_optional_model_names(self) -> list:
        """未ロードのモデル名リストを取得"""
        return [name for name in self.get_all_model_names() if name not in self.loaded_models]
    
    def is_loaded(self, model_name: str) -> bool:
        """モデルがロード済みかチェック"""
        return model_name in self.loaded_models
    
    def get_model(self, model_name: str):
        """ロード済みモデルを取得"""
        return self.loaded_models.get(model_name)
    
    def load_model_sync(self, model_name: str):
        """モデルを同期的にロード"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        ModelClass = self.ALL_MODEL_CLASSES.get(model_name)
        if ModelClass is None:
            raise ValueError(f"Model class not found for: {model_name}")
        
        model_instance = ModelClass(weights='imagenet')
        self.loaded_models[model_name] = model_instance
        logging.info(f"Model {model_name} loaded successfully.")
        return model_instance
    
    def load_model_async(self, model_name: str):
        """モデルを非同期でロード (Futureを返す)"""
        if model_name in self.loaded_models:
            return None
        
        def _load():
            ModelClass = self.ALL_MODEL_CLASSES.get(model_name)
            if ModelClass is None:
                raise ValueError(f"Model class not found for: {model_name}")
            model_instance = ModelClass(weights='imagenet')
            return model_name, model_instance
        
        return self.executor.submit(_load)
    
    def register_loaded_model(self, model_name: str, model_instance):
        """ロード完了したモデルを登録"""
        self.loaded_models[model_name] = model_instance
        logging.info(f"Model {model_name} registered successfully.")
    
    def load_initial_models(self):
        """初期モデルをロード"""
        for name in self.INITIAL_MODELS:
            if name not in self.loaded_models:
                self.load_model_sync(name)
    
    def shutdown(self):
        """エグゼキューターをシャットダウン"""
        self.executor.shutdown(wait=False)
