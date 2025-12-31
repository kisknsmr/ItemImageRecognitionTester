"""
ウィジェットモジュール
画像・テキスト表示用のヘルパー関数
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def display_image(frame, row: int, col: int, img_path: str, 
                  colspan: int = 1, size: tuple = (100, 100), 
                  padx: int = 2, pady: int = 2) -> ttk.Label:
    """
    画像をフレームに表示
    
    Args:
        frame: 親フレーム
        row: 行位置
        col: 列位置
        img_path: 画像パス
        colspan: 列スパン
        size: サムネイルサイズ
        padx: X方向パディング
        pady: Y方向パディング
        
    Returns:
        作成したラベルウィジェット
    """
    pil_img = Image.open(img_path)
    pil_img.thumbnail(size)
    tk_img = ImageTk.PhotoImage(pil_img)
    
    label = ttk.Label(frame, image=tk_img)
    label.image = tk_img  # 参照保持
    label.grid(row=row, column=col, columnspan=colspan, sticky='nsew', padx=padx, pady=pady)
    return label


def display_text(frame, row: int, col: int, text: str,
                 colspan: int = 1, padx: int = 2, pady: int = 2,
                 font: tuple = ("Helvetica", 8)) -> ttk.Label:
    """
    テキストをフレームに表示
    
    Args:
        frame: 親フレーム
        row: 行位置
        col: 列位置
        text: 表示テキスト
        colspan: 列スパン
        padx: X方向パディング
        pady: Y方向パディング
        font: フォント設定
        
    Returns:
        作成したラベルウィジェット
    """
    label = ttk.Label(frame, text=text, justify=tk.LEFT, anchor="nw", font=font)
    label.grid(row=row, column=col, columnspan=colspan, sticky='nsew', padx=padx, pady=pady)
    return label


def configure_ref_item(ref_frame, img_path: str, result_text: str):
    """
    参照画像をフレームに配置
    
    Args:
        ref_frame: 参照画像用フレーム
        img_path: 画像パス
        result_text: 結果テキスト
    """
    display_image(ref_frame, 0, 0, img_path, colspan=1, size=(200, 200))
    display_text(ref_frame, 0, 1, result_text, colspan=1)


def configure_options_item(options_frame, index: int, img_path: str, 
                           result_text: str, is_highlighted: bool = False):
    """
    オプション画像をフレームに配置（ハイライト対応）
    
    Args:
        options_frame: オプション画像用フレーム
        index: 画像インデックス (1-9)
        img_path: 画像パス
        result_text: 結果テキスト
        is_highlighted: ハイライト表示するか
    """
    relative_index = index - 1
    row = relative_index // 3
    col_img = 2 * (relative_index % 3)
    
    highlight_style = "Highlight.TFrame" if is_highlighted else "TFrame"
    
    panel_frame = ttk.Frame(
        options_frame, 
        style=highlight_style, 
        borderwidth=3,
        relief="raised" if is_highlighted else "flat"
    )
    panel_frame.grid(row=row, column=col_img, columnspan=2, sticky='nsew', padx=5, pady=5)
    panel_frame.grid_rowconfigure(0, weight=1)
    panel_frame.grid_columnconfigure(0, weight=1)
    panel_frame.grid_columnconfigure(1, weight=3)
    
    display_image(panel_frame, 0, 0, img_path, size=(80, 80), padx=5)
    display_text(panel_frame, 0, 1, result_text, colspan=1, padx=5)
