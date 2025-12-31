"""
レイアウトモジュール
UIレイアウトの構築を担当
"""
import tkinter as tk
from tkinter import ttk


class LayoutManager:
    """UIレイアウト管理クラス"""
    
    def __init__(self, app):
        """
        Args:
            app: メインアプリケーションインスタンス
        """
        self.app = app
    
    def configure_style(self):
        """スタイル設定"""
        self.app.style = ttk.Style()
        self.app.style.theme_use('clam')
        self.app.style.configure("Highlight.TFrame", background="#E0FFFF")
    
    def setup_layout(self):
        """メインレイアウトを構築"""
        # メインフレーム
        main_frame = ttk.Frame(self.app)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # 左フレーム（画像表示用）
        self._setup_left_frame(main_frame)
        
        # 右フレーム（コントロール）
        self._setup_right_frame(main_frame)
        
        # ステータスバー
        self._setup_status_bar()
    
    def _setup_left_frame(self, parent):
        """左側フレーム（画像表示エリア）を構築"""
        self.app.left_frame = ttk.Frame(parent, relief="flat")
        self.app.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.app.left_frame.grid_rowconfigure(0, weight=1)
        self.app.left_frame.grid_rowconfigure(1, weight=3)
        self.app.left_frame.grid_columnconfigure(0, weight=1)
        
        # 参照画像フレーム
        self.app.ref_frame = ttk.Frame(self.app.left_frame, borderwidth=1, relief="solid")
        self.app.ref_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.app.ref_frame.grid_rowconfigure(0, weight=1)
        self.app.ref_frame.grid_columnconfigure(0, weight=1)
        self.app.ref_frame.grid_columnconfigure(1, weight=1)
        
        # オプション画像フレーム
        self.app.options_frame = ttk.Frame(self.app.left_frame, borderwidth=1, relief="flat")
        self.app.options_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        for i in range(3):
            self.app.options_frame.grid_rowconfigure(i, weight=1)
        for i in range(6):
            self.app.options_frame.grid_columnconfigure(i, weight=1)
    
    def _setup_right_frame(self, parent):
        """右側フレーム（コントロールエリア）を構築"""
        self.app.right_frame = ttk.Frame(parent, borderwidth=1, relief="flat")
        self.app.right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.app.right_frame.grid_columnconfigure(0, weight=1)
        self.app.right_frame.grid_rowconfigure(2, weight=1)
        self.app.right_frame.grid_rowconfigure(4, weight=1)
        
        # モデル表示ラベル
        self.app.model_label = ttk.Label(
            self.app.right_frame, 
            text=f"Selected Model: {self.app.model_name}"
        )
        self.app.model_label.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # フォルダ選択ボタン
        ttk.Button(
            self.app.right_frame, 
            text="画像フォルダ変更", 
            command=self.app.select_folder
        ).grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # ロード済みモデルセクション
        ttk.Label(
            self.app.right_frame, 
            text="--- Loaded Models ---", 
            anchor="center"
        ).grid(row=2, column=0, sticky="ew", padx=5, pady=(10, 0))
        
        self.app.model_frame = ttk.Frame(self.app.right_frame)
        self.app.model_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.app.model_frame.grid_columnconfigure(0, weight=1)
        
        # オプションモデルセクション
        ttk.Label(
            self.app.right_frame, 
            text="--- Optional Models (Click to Load) ---", 
            anchor="center"
        ).grid(row=4, column=0, sticky="ew", padx=5, pady=(10, 0))
        
        self.app.optional_frame = ttk.Frame(self.app.right_frame)
        self.app.optional_frame.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)
        self.app.optional_frame.grid_columnconfigure(0, weight=1)
    
    def _setup_status_bar(self):
        """ステータスバーを構築"""
        self.app.status_bar = tk.Frame(self.app, bd=1, relief=tk.SUNKEN)
        self.app.status_bar.grid(row=1, column=0, sticky="ew")
        self.app.status_bar.grid_columnconfigure(0, weight=1)
        
        self.app.status_label = ttk.Label(self.app.status_bar, text="Ready", anchor="w")
        self.app.status_label.grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        
        self.app.progress = ttk.Progressbar(
            self.app.status_bar, 
            orient="horizontal", 
            length=200, 
            mode="determinate"
        )
        self.app.progress.grid(row=0, column=1, padx=10, pady=2, sticky="e")
