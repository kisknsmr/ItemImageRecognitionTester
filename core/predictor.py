"""
画像予測モジュール
画像の前処理・推論・セマンティックマッチングを担当する
"""
import traceback
import numpy as np
import logging
from tensorflow.keras.preprocessing import image


class ImagePredictor:
    """画像予測・セマンティックマッチングクラス"""
    
    def __init__(self, model_manager):
        """
        Args:
            model_manager: ModelManagerインスタンス
        """
        self.model_manager = model_manager
    
    def get_reference_identity(self, model_name: str, model, img_path: str):
        """
        参照画像のTop-10クラス名を取得（同期処理）
        
        Args:
            model_name: モデル名
            model: ロード済みモデルインスタンス
            img_path: 画像パス
            
        Returns:
            tuple: (クラス名リスト, 予測テキスト) または ([], エラーメッセージ)
        """
        try:
            config = self.model_manager.get_model_config(model_name)
            if config is None:
                return [], f"Model config not found for: {model_name}"
            
            preprocess_func, target_size, decode_func = config
            
            img = image.load_img(img_path, target_size=target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_func(x)
            
            preds = model.predict(x, verbose=0)
            top_preds = decode_func(preds, top=10)[0]
            
            ref_class_names = [pred[1] for pred in top_preds]
            
            predictions_text = (
                f"TARGET IDs (Top 10):\n{', '.join(ref_class_names)}\n\n"
                f"Top 10 Predictions:\n" +
                "\n".join([f"{pred[1]} ({pred[2]:.4f})" for pred in top_preds])
            )
            
            return ref_class_names, predictions_text
        
        except Exception as e:
            logging.error(f"Error in get_reference_identity: {e}")
            return [], f"Error determining IDs: {e}"
    
    def run_semantic_matching(self, model_name: str, model, img_path: str, ref_class_names: list):
        """
        セマンティックマッチング予測（バックグラウンドスレッド用）
        
        Args:
            model_name: モデル名
            model: ロード済みモデルインスタンス
            img_path: 画像パス
            ref_class_names: 参照画像のクラス名リスト
            
        Returns:
            tuple: (img_path, result_text, is_match, confidence, rank)
        """
        try:
            config = self.model_manager.get_model_config(model_name)
            if config is None:
                return img_path, f"Model config not found", False, 0.0, 11
            
            preprocess_func, target_size, decode_func = config
            
            img = image.load_img(img_path, target_size=target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_func(x)
            
            preds = model.predict(x, verbose=0)
            top_preds = decode_func(preds, top=10)[0]
            
            is_match = False
            best_match_class = "None"
            best_match_confidence = 0.0
            rank_of_match = 11  # 10位以下を意味
            
            option_top_names_conf = {pred[1]: pred[2] for pred in top_preds}
            
            # 参照画像のTop10と照合
            for rank, target_name in enumerate(ref_class_names):
                if target_name in option_top_names_conf:
                    is_match = True
                    best_match_class = target_name
                    best_match_confidence = option_top_names_conf[target_name]
                    rank_of_match = rank + 1
                    break
            
            predictions_text = (
                f"MATCH: {is_match}\n"
                f"Confidence: {best_match_confidence:.4f}\n"
                f"Matched ID: {best_match_class}\n"
                f"Ref Rank: {rank_of_match}\n\n"
                f"Top 10 Predictions:\n" +
                "\n".join([f"{pred[1]} ({pred[2]:.4f})" for pred in top_preds])
            )
            
            return img_path, predictions_text, is_match, best_match_confidence, rank_of_match
        
        except Exception as e:
            error_info = traceback.format_exc()
            logging.error(f"Prediction error for {img_path}: {error_info}")
            return img_path, f"Prediction Error: {type(e).__name__}: {e}", False, 0.0, 11
    
    @staticmethod
    def select_top_results(raw_results: list, top_n: int = 3) -> list:
        """
        予測結果から上位N件を選定
        
        Args:
            raw_results: 予測結果リスト (dict形式)
            top_n: 選定件数
            
        Returns:
            上位N件のインデックスリスト
        """
        sorted_results = sorted(
            raw_results,
            key=lambda x: (x['match'], x['confidence'], -x['ref_rank']),
            reverse=True
        )
        return [item['index'] for item in sorted_results[:top_n]]
