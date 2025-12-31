import traceback
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from concurrent.futures import ThreadPoolExecutor
import sys
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- TensorFlowモデルのインポート (全モデル) ---
from tensorflow.keras.applications import (
    ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2,
    VGG16, VGG19, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, Xception,
    DenseNet121, DenseNet169, DenseNet201, NASNetMobile, NASNetLarge,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4,
    EfficientNetB5, EfficientNetB6, EfficientNetB7,
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2,
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet_v2
from tensorflow.keras.applications import (
    vgg16, vgg19, inception_v3, inception_resnet_v2, mobilenet, mobilenet_v2, xception,
    densenet, nasnet, efficientnet,
)


class ImageRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("深層学習画像認識比較アプリ (選定ハイライト版)")
        self.geometry("1400x900")

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.img_dir = os.path.join(os.getcwd(), "ReferencePicture", "9panel3")
        self.model_name = "ResNet50"
        self.current_model = None
        self.is_processing = False

        self.loaded_models = {}
        self.raw_results = []  # 全てのオプションの結果を保持 (選定用)

        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

        self.setup_model_map()
        self.configure_style()
        self.setup_layout()
        self.initial_load()
        self.predict_and_display_images()

    def setup_model_map(self):
        # --- モデル設定マップ (変更なし、省略) ---
        size_map = {"224": (224, 224), "299": (299, 299), "331": (331, 331), "600": (600, 600),
                    "240": (240, 240), "260": (260, 260), "300": (300, 300), "380": (380, 380),
                    "456": (456, 456), "528": (528, 528), }
        self.initial_models = ["ResNet50", "VGG16", "MobileNetV2", "EfficientNetB0", "Xception"]
        self.all_models_classes = {
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
        self.model_config = {
            "ResNet50": (preprocess_resnet50, size_map["224"], decode_predictions),
            "VGG16": (vgg16.preprocess_input, size_map["224"], decode_predictions),
            "MobileNetV2": (mobilenet_v2.preprocess_input, size_map["224"], decode_predictions),
            "EfficientNetB0": (efficientnet.preprocess_input, size_map["224"], decode_predictions),
            "Xception": (xception.preprocess_input, size_map["299"], decode_predictions),
            "ResNet101": (preprocess_resnet, size_map["224"], decode_predictions),
            "EfficientNetV2B1": (efficientnet.preprocess_input, size_map["240"], decode_predictions),
            "NASNetLarge": (nasnet.preprocess_input, size_map["331"], decode_predictions),
            "DenseNet121": (densenet.preprocess_input, size_map["224"], decode_predictions),
        }

    def configure_style(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Highlight.TFrame", background="#E0FFFF")  # 選定ハイライト用スタイル

    def setup_layout(self):
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # 左フレーム（画像表示用）
        self.left_frame = ttk.Frame(main_frame, relief="flat")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.left_frame.grid_rowconfigure(0, weight=1)  # 参照フレーム
        self.left_frame.grid_rowconfigure(1, weight=3)  # オプションフレーム
        self.left_frame.grid_columnconfigure(0, weight=1)

        # --- レイアウト分離 ---
        # 参照画像フレーム (上段)
        self.ref_frame = ttk.Frame(self.left_frame, borderwidth=1, relief="solid")
        self.ref_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.ref_frame.grid_rowconfigure(0, weight=1)
        self.ref_frame.grid_columnconfigure(0, weight=1)
        self.ref_frame.grid_columnconfigure(1, weight=1)

        # オプション画像フレーム (下段)
        self.options_frame = ttk.Frame(self.left_frame, borderwidth=1, relief="flat")
        self.options_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        # 3x3のグリッドの重み付けを事前に設定
        for i in range(3):
            self.options_frame.grid_rowconfigure(i, weight=1)
        for i in range(6):
            self.options_frame.grid_columnconfigure(i, weight=1)
        # ---------------------

        # 右フレーム（コントロール）
        self.right_frame = ttk.Frame(main_frame, borderwidth=1, relief="flat")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(2, weight=1)
        self.right_frame.grid_rowconfigure(4, weight=1)

        # 1. 選択モデル表示
        self.model_label = ttk.Label(self.right_frame, text=f"Selected Model: {self.model_name}")
        self.model_label.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # 2. フォルダ選択ボタン
        ttk.Button(self.right_frame, text="画像フォルダ変更", command=self.select_folder).grid(
            row=1, column=0, sticky="ew", padx=5, pady=5)

        # 3. ロード済みモデルボタン配置用フレーム
        ttk.Label(self.right_frame, text="--- Loaded Models ---", anchor="center").grid(row=2, column=0, sticky="ew",
                                                                                        padx=5, pady=(10, 0))
        self.model_frame = ttk.Frame(self.right_frame)
        self.model_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.model_frame.grid_columnconfigure(0, weight=1)

        # 4. オプションモデル配置用フレーム
        ttk.Label(self.right_frame, text="--- Optional Models (Click to Load) ---", anchor="center").grid(row=4,
                                                                                                          column=0,
                                                                                                          sticky="ew",
                                                                                                          padx=5,
                                                                                                          pady=(10, 0))
        self.optional_frame = ttk.Frame(self.right_frame)
        self.optional_frame.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)
        self.optional_frame.grid_columnconfigure(0, weight=1)

        # 5. ステータスバー（変更なし）
        self.status_bar = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        self.status_bar.grid(row=1, column=0, sticky="ew")
        self.status_bar.grid_columnconfigure(0, weight=1)

        self.status_label = ttk.Label(self.status_bar, text="Ready", anchor="w")
        self.status_label.grid(row=0, column=0, padx=5, pady=2, sticky="ew")

        self.progress = ttk.Progressbar(self.status_bar, orient="horizontal", length=200, mode="determinate")
        self.progress.grid(row=0, column=1, padx=10, pady=2, sticky="e")

    def initial_load(self):
        self.model_name = "ResNet50"

        for name in self.initial_models:
            self._load_model_by_name(name)

        self.current_model = self.loaded_models.get(self.model_name)

        self.create_model_buttons()

    def _load_model_by_name(self, model_name):
        if model_name in self.loaded_models:
            return True

        self.set_status(f"Loading {model_name} asynchronously...")

        future = self.executor.submit(self._async_load_model, model_name)

        self.after(100, lambda: self._check_single_load_result(future, model_name))

        return False

    def _async_load_model(self, model_name):

        ModelClass = self.all_models_classes.get(model_name)

        if ModelClass is None:
            raise ValueError(f"Model class not found for: {model_name}")

        model_instance = ModelClass(weights='imagenet')
        return model_name, model_instance

    def _check_single_load_result(self, future, model_name):
        if future.done():
            try:
                name, instance = future.result()
                self.loaded_models[name] = instance

                self.set_status(f"Model {name} loaded successfully.")
                self.create_model_buttons()

                if name == self.model_name:
                    self.current_model = instance
                    self.predict_and_display_images()

            except Exception as e:
                self.set_status(f"Error loading model {model_name}: {e}")
                logging.error(f"Error loading {model_name}: {e}")
        else:
            self.after(100, lambda: self._check_single_load_result(future, model_name))

    def create_model_buttons(self):

        for widget in self.model_frame.winfo_children():
            widget.destroy()

        for widget in self.optional_frame.winfo_children():
            widget.destroy()

        loaded_names = sorted(list(self.loaded_models.keys()))
        all_names = sorted(list(self.all_models_classes.keys()))

        max_name_length = max(len(name) for name in all_names) if all_names else 10

        # 1. ロード済みモデルボタン
        for idx, name in enumerate(loaded_names):
            btn = ttk.Button(self.model_frame, text=name,
                             command=lambda name=name: self.change_model_and_predict(name))
            btn.grid(row=idx // 3, column=idx % 3, padx=3, pady=3, sticky="ew")

        # 2. オプションモデルボタン
        optional_names = [name for name in all_names if name not in self.loaded_models]

        for idx, name in enumerate(optional_names):
            btn = ttk.Button(self.optional_frame, text=name,
                             command=lambda name=name: self.load_optional_model_and_switch(name))
            btn.grid(row=idx // 3, column=idx % 3, padx=3, pady=3, sticky="ew")

        # フレームの列幅設定
        for frame in [self.model_frame, self.optional_frame]:
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_columnconfigure(1, weight=1)
            frame.grid_columnconfigure(2, weight=1)

    def load_optional_model_and_switch(self, new_model_name):

        if new_model_name in self.loaded_models:
            self.change_model_and_predict(new_model_name)
            return

        self.set_status(f"Loading optional model: {new_model_name}...")
        future = self.executor.submit(self._async_load_model, new_model_name)

        def switch_after_load(future):
            if future.done():
                try:
                    name, instance = future.result()
                    self.loaded_models[name] = instance
                    self.set_status(f"Model {name} loaded. Switching...")
                    self.create_model_buttons()

                    self.change_model_and_predict(name)

                except Exception as e:
                    self.set_status(f"Error loading {new_model_name}: {e}")
                    logging.error(f"Error loading {new_model_name}: {e}")

        self.after(100, lambda: self._check_future_for_callback(future, switch_after_load))

    def _check_future_for_callback(self, future, callback):
        if future.done():
            callback(future)
        else:
            self.after(100, lambda: self._check_future_for_callback(future, callback))

    def predict_and_display_images(self):
        # ... (予測ロジックの準備) ...
        if self.is_processing or not self.current_model:
            if not self.current_model:
                self.set_status(f"Error: Model {self.model_name} not loaded. Cannot predict.")
            return

        self.is_processing = True
        self.set_status("Determining reference identity...")
        self.progress['value'] = 0

        # 既存のウィジェットをクリア (ref_frameとoptions_frameのすべて)
        for widget in self.ref_frame.winfo_children(): widget.destroy()
        for widget in self.options_frame.winfo_children(): widget.destroy()

        self.raw_results = []  # 結果リストをリセット

        img_files = ["0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png"]
        ref_path = os.path.join(self.img_dir, img_files[0])

        if not os.path.exists(ref_path):
            self.set_status(f"Error: Reference image {img_files[0]} not found in {self.img_dir}.")
            self.is_processing = False
            return

        # 1. 参照画像のIDを同期的に決定
        ref_ids, ref_preds = self._get_reference_identity(ref_path)

        if not ref_ids:
            self.set_status(ref_preds)
            self.is_processing = False
            return

        # 参照画像を表示 (ref_frameに配置)
        self.configure_ref_item(ref_path, ref_preds)
        self.set_status(f"Target IDs determined: {len(ref_ids)} IDs. Starting semantic matching...")

        # 2. オプションの画像を非同期で予測 (セマンティックマッチング)
        option_files = img_files[1:]
        total = len(option_files)
        futures = []

        for i, img_file in enumerate(option_files):
            img_path = os.path.join(self.img_dir, img_file)

            if os.path.exists(img_path):
                # Submit prediction task with the set of target reference IDs
                future = self.executor.submit(self._run_semantic_matching_task, img_path, ref_ids)
                # i+1 は元のインデックス (1, 2, ..., 9)
                futures.append((future, i + 1, img_path))
            else:
                self.set_status(f"Warning: Image {img_file} not found.")

        self._check_prediction_results(futures, total)

    def _get_reference_identity(self, img_path):
        """Reference image processing to get the Top-10 target class names (Synchronous)"""
        try:
            config = self.model_config.get(self.model_name)
            preprocess_func, target_size, decode_func = config

            img = image.load_img(img_path, target_size=target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_func(x)

            preds = self.current_model.predict(x, verbose=0)
            top_preds = decode_func(preds, top=10)[0]

            ref_class_names = [pred[1] for pred in top_preds]

            predictions_text = f"TARGET IDs (Top 10):\n{', '.join(ref_class_names)}\n\nTop 10 Predictions:\n" + \
                               "\n".join([f"{pred[1]} ({pred[2]:.4f})" for pred in top_preds])

            return ref_class_names, predictions_text

        except Exception as e:
            logging.error(f"Error in _get_reference_identity: {e}")
            return [], f"Error determining IDs: {e}"

    def _run_semantic_matching_task(self, img_path, ref_class_names):
        """Background thread semantic matching prediction."""
        try:
            config = self.model_config.get(self.model_name)
            preprocess_func, target_size, decode_func = config

            img = image.load_img(img_path, target_size=target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_func(x)

            preds = self.current_model.predict(x, verbose=0)
            top_preds = decode_func(preds, top=10)[0]

            is_match = False
            best_match_class = "None"
            best_match_confidence = 0.0
            rank_of_match = 11  # 10位以下を意味

            option_top_names_conf = {pred[1]: pred[2] for pred in top_preds}

            # Check if any of the option's Top 10 predictions match any of the reference's Top 10 IDs
            for rank, target_name in enumerate(ref_class_names):  # ref_class_namesはランク順(0-9)
                if target_name in option_top_names_conf:
                    is_match = True
                    best_match_class = target_name
                    best_match_confidence = option_top_names_conf[target_name]
                    rank_of_match = rank + 1  # 1位が1
                    break

            # Build prediction text for display
            predictions_text = f"MATCH: {is_match}\n" + \
                               f"Confidence: {best_match_confidence:.4f}\n" + \
                               f"Matched ID: {best_match_class}\n" + \
                               f"Ref Rank: {rank_of_match}\n\n" + \
                               "Top 10 Predictions:\n" + \
                               "\n".join([f"{pred[1]} ({pred[2]:.4f})" for pred in top_preds])

            # 選定ロジックのために結果を返す
            return img_path, predictions_text, is_match, best_match_confidence, rank_of_match

        except Exception as e:
            error_info = traceback.format_exc()
            return img_path, f"Prediction Error: {type(e).__name__}: {e}", False, 0.0, 11

    def _check_prediction_results(self, futures, total_tasks, completed_count=0):

        newly_completed = []

        for future, index, path in futures:
            if future.done():
                newly_completed.append((future, index, path))

        if newly_completed:
            futures[:] = [(f, i, p) for f, i, p in futures if not f.done()]

            for future, index, path in newly_completed:
                completed_count += 1
                # (path, text, is_match, confidence, rank) のタプルを取得
                img_path, result_text, is_match, confidence, rank = future.result()

                # 選定ロジック用に結果をリストに保持
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

            # --- ここで選定ロジックを実行し、GUIを最終更新 ---
            self._finalize_options_display()
            # ----------------------------------------------

    def _finalize_options_display(self):
        """全予測完了後、選定ロジックに基づいて上位3つを決定し、GUIを更新する"""

        # 1. 確信度 (confidence) と 参照ランク (ref_rank) に基づいてソート
        # 選定基準: マッチング成功 (True) を優先し、その中で確信度が高い順、次に参照ランクが高いIDとのマッチを優先。
        sorted_results = sorted(self.raw_results,
                                key=lambda x: (x['match'], x['confidence'], -x['ref_rank']),
                                reverse=True)

        # 2. 上位3つのインデックスを特定
        top_3_indices = [item['index'] for item in sorted_results[:3]]

        # 3. GUIの再配置とハイライト
        for result in self.raw_results:
            is_highlighted = result['index'] in top_3_indices
            self.configure_options_item(result['index'], result['path'], result['text'], is_highlighted)

        self.set_status(f"Analysis complete. Top 3 results highlighted based on confidence.")
        self.raw_results = []  # リセット

    def configure_ref_item(self, img_path, result_text):
        """参照画像をref_frameに配置する (index=0)"""

        # 参照画像（左側）
        self.display_image(0, 0, img_path, colspan=1, in_frame=self.ref_frame, size=(200, 200))
        # 参照テキスト（右側）
        self.display_text(0, 1, result_text, in_frame=self.ref_frame, colspan=1)

    def configure_options_item(self, index, img_path, result_text, is_highlighted=False):
        """オプション画像をoptions_frameに配置し、ハイライトを適用する"""

        relative_index = index - 1
        row = relative_index // 3
        col_img = 2 * (relative_index % 3)
        col_text = col_img + 1

        # ハイライト用のフレームを画像とテキストの外側に作成
        highlight_style = "Highlight.TFrame" if is_highlighted else "TFrame"

        panel_frame = ttk.Frame(self.options_frame, style=highlight_style, borderwidth=3,
                                relief="raised" if is_highlighted else "flat")
        panel_frame.grid(row=row, column=col_img, columnspan=2, sticky='nsew', padx=5, pady=5)
        panel_frame.grid_rowconfigure(0, weight=1)
        panel_frame.grid_columnconfigure(0, weight=1)
        panel_frame.grid_columnconfigure(1, weight=3)  # 画像とテキストの相対的な幅

        # 画像
        self.display_image(0, 0, img_path, in_frame=panel_frame, size=(80, 80), padx=5)
        # テキスト
        self.display_text(0, 1, result_text, in_frame=panel_frame, colspan=1, padx=5)

    def display_image(self, row, col, img_path, colspan=1, in_frame=None, size=(100, 100), padx=2, pady=2):
        pil_img = Image.open(img_path)
        pil_img.thumbnail(size)
        tk_img = ImageTk.PhotoImage(pil_img)

        label = ttk.Label(in_frame, image=tk_img)
        label.image = tk_img
        label.grid(row=row, column=col, columnspan=colspan, sticky='nsew', padx=padx, pady=pady)

    def display_text(self, row, col, text, colspan=1, in_frame=None, padx=2, pady=2):
        label = ttk.Label(in_frame, text=text, justify=tk.LEFT, anchor="nw", font=("Helvetica", 8))
        label.grid(row=row, column=col, columnspan=colspan, sticky='nsew', padx=padx, pady=pady)

    def change_model_and_predict(self, new_model_name):
        if self.is_processing:
            self.set_status("Wait for the current process to complete.")
            return

        if new_model_name == self.model_name:
            return

        if new_model_name not in self.loaded_models:
            self.set_status(f"Error: Model {new_model_name} not loaded.")
            return

        self.model_name = new_model_name
        self.model_label.config(text=f"Selected Model: {self.model_name}")
        self.current_model = self.loaded_models.get(new_model_name)

        self.predict_and_display_images()

    def set_status(self, message):
        self.status_label.config(text=message)

    def display_error(self):
        pass

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="画像フォルダを選択してください")
        if folder_path:
            self.img_dir = folder_path
            self.set_status(f"Selected folder: {self.img_dir}")
            self.predict_and_display_images()


if __name__ == "__main__":
    app = ImageRecognitionApp()
    app.mainloop()