"""
メインアプリケーションモジュール
ImageRecognitionAppクラスを定義
"""
import os
import logging
import tkinter as tk
from tkinter import ttk, filedialog

from core.model_manager import ModelManager
from core.predictor import ImagePredictor
from gui.layouts import LayoutManager
from gui import widgets

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageRecognitionApp(tk.Tk):
    """深層学習画像認識比較アプリケーション"""
    
    def __init__(self):
        super().__init__()
        
        self.title("深層学習画像認識比較アプリ (選定ハイライト版)")
        self.geometry("1400x900")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)
        
        # 状態変数
        self.img_dir = os.path.join(os.getcwd(), "ReferencePicture", "9panel3")
        self.model_name = "ResNet50"
        self.current_model = None
        self.is_processing = False
        self.raw_results = []
        
        # マネージャー初期化
        self.model_manager = ModelManager()
        self.predictor = ImagePredictor(self.model_manager)
        self.layout_manager = LayoutManager(self)
        
        # UI構築
        self.layout_manager.configure_style()
        self.layout_manager.setup_layout()
        
        # 初期化
        self._initial_load()
        self.predict_and_display_images()
    
    def _initial_load(self):
        """初期モデルをロード"""
        self.model_manager.load_initial_models()
        self.current_model = self.model_manager.get_model(self.model_name)
        self._create_model_buttons()
    
    def _create_model_buttons(self):
        """モデルボタンを作成"""
        # 既存ボタンをクリア
        for widget in self.model_frame.winfo_children():
            widget.destroy()
        for widget in self.optional_frame.winfo_children():
            widget.destroy()
        
        loaded_names = self.model_manager.get_loaded_model_names()
        optional_names = self.model_manager.get_optional_model_names()
        
        # ロード済みモデルボタン
        for idx, name in enumerate(loaded_names):
            btn = ttk.Button(
                self.model_frame, 
                text=name,
                command=lambda n=name: self._change_model_and_predict(n)
            )
            btn.grid(row=idx // 3, column=idx % 3, padx=3, pady=3, sticky="ew")
        
        # オプションモデルボタン
        for idx, name in enumerate(optional_names):
            btn = ttk.Button(
                self.optional_frame, 
                text=name,
                command=lambda n=name: self._load_optional_model_and_switch(n)
            )
            btn.grid(row=idx // 3, column=idx % 3, padx=3, pady=3, sticky="ew")
        
        # 列幅設定
        for frame in [self.model_frame, self.optional_frame]:
            for i in range(3):
                frame.grid_columnconfigure(i, weight=1)
    
    def _load_optional_model_and_switch(self, new_model_name: str):
        """オプションモデルをロードして切り替え"""
        if self.model_manager.is_loaded(new_model_name):
            self._change_model_and_predict(new_model_name)
            return
        
        self.set_status(f"Loading optional model: {new_model_name}...")
        future = self.model_manager.load_model_async(new_model_name)
        
        if future is None:
            return
        
        def check_and_switch():
            if future.done():
                try:
                    name, instance = future.result()
                    self.model_manager.register_loaded_model(name, instance)
                    self.set_status(f"Model {name} loaded. Switching...")
                    self._create_model_buttons()
                    self._change_model_and_predict(name)
                except Exception as e:
                    self.set_status(f"Error loading {new_model_name}: {e}")
                    logging.error(f"Error loading {new_model_name}: {e}")
            else:
                self.after(100, check_and_switch)
        
        self.after(100, check_and_switch)
    
    def _change_model_and_predict(self, new_model_name: str):
        """モデルを切り替えて予測実行"""
        if self.is_processing:
            self.set_status("Wait for the current process to complete.")
            return
        
        if new_model_name == self.model_name:
            return
        
        if not self.model_manager.is_loaded(new_model_name):
            self.set_status(f"Error: Model {new_model_name} not loaded.")
            return
        
        self.model_name = new_model_name
        self.model_label.config(text=f"Selected Model: {self.model_name}")
        self.current_model = self.model_manager.get_model(new_model_name)
        
        self.predict_and_display_images()
    
    def predict_and_display_images(self):
        """画像の予測と表示を実行"""
        if self.is_processing or not self.current_model:
            if not self.current_model:
                self.set_status(f"Error: Model {self.model_name} not loaded. Cannot predict.")
            return
        
        self.is_processing = True
        self.set_status("Determining reference identity...")
        self.progress['value'] = 0
        
        # ウィジェットをクリア
        for widget in self.ref_frame.winfo_children():
            widget.destroy()
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        
        self.raw_results = []
        
        img_files = [f"{i}.png" for i in range(10)]
        ref_path = os.path.join(self.img_dir, img_files[0])
        
        if not os.path.exists(ref_path):
            self.set_status(f"Error: Reference image {img_files[0]} not found in {self.img_dir}.")
            self.is_processing = False
            return
        
        # 参照画像のID決定
        ref_ids, ref_preds = self.predictor.get_reference_identity(
            self.model_name, self.current_model, ref_path
        )
        
        if not ref_ids:
            self.set_status(ref_preds)
            self.is_processing = False
            return
        
        # 参照画像を表示
        widgets.configure_ref_item(self.ref_frame, ref_path, ref_preds)
        self.set_status(f"Target IDs determined: {len(ref_ids)} IDs. Starting semantic matching...")
        
        # オプション画像の予測
        option_files = img_files[1:]
        total = len(option_files)
        futures = []
        
        for i, img_file in enumerate(option_files):
            img_path = os.path.join(self.img_dir, img_file)
            
            if os.path.exists(img_path):
                future = self.model_manager.executor.submit(
                    self.predictor.run_semantic_matching,
                    self.model_name, self.current_model, img_path, ref_ids
                )
                futures.append((future, i + 1, img_path))
            else:
                self.set_status(f"Warning: Image {img_file} not found.")
        
        self._check_prediction_results(futures, total)
    
    def _check_prediction_results(self, futures, total_tasks, completed_count=0):
        """予測結果をチェック"""
        newly_completed = [(f, i, p) for f, i, p in futures if f.done()]
        
        if newly_completed:
            futures[:] = [(f, i, p) for f, i, p in futures if not f.done()]
            
            for future, index, path in newly_completed:
                completed_count += 1
                img_path, result_text, is_match, confidence, rank = future.result()
                
                self.raw_results.append({
                    'index': index,
                    'path': img_path,
                    'text': result_text,
                    'match': is_match,
                    'confidence': confidence,
                    'ref_rank': rank
                })
                
                self.progress['value'] = (completed_count / total_tasks) * 100
                self.set_status(f"Processing... ({completed_count}/{total_tasks} completed)")
        
        if futures:
            self.after(100, lambda: self._check_prediction_results(futures, total_tasks, completed_count))
        else:
            self.is_processing = False
            self.set_status(f"Completed! {self.model_name} predictions displayed. Identifying Top 3...")
            self.progress['value'] = 100
            self._finalize_options_display()
    
    def _finalize_options_display(self):
        """結果表示を最終化"""
        top_3_indices = ImagePredictor.select_top_results(self.raw_results, top_n=3)
        
        for result in self.raw_results:
            is_highlighted = result['index'] in top_3_indices
            widgets.configure_options_item(
                self.options_frame, 
                result['index'], 
                result['path'], 
                result['text'], 
                is_highlighted
            )
        
        self.set_status("Analysis complete. Top 3 results highlighted based on confidence.")
        self.raw_results = []
    
    def set_status(self, message: str):
        """ステータスバーを更新"""
        self.status_label.config(text=message)
    
    def select_folder(self):
        """フォルダ選択ダイアログ"""
        folder_path = filedialog.askdirectory(title="画像フォルダを選択してください")
        if folder_path:
            self.img_dir = folder_path
            self.set_status(f"Selected folder: {self.img_dir}")
            self.predict_and_display_images()
    
    def destroy(self):
        """アプリケーション終了時のクリーンアップ"""
        self.model_manager.shutdown()
        super().destroy()
