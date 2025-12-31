"""
画像分割ユーティリティ
画像を3x3の9パネルに分割する機能
"""
import os
import datetime
from PIL import Image


def split_image_9panels(image_path: str, output_dir: str = None, 
                        temp_dir: str = "temppicture/9panel") -> list:
    """
    画像を3x3の9パネルに分割
    
    Args:
        image_path: 入力画像のパス
        output_dir: 出力ディレクトリ（Noneの場合はlearningdata）
        temp_dir: 一時保存ディレクトリ
        
    Returns:
        生成されたファイルパスのリスト
    """
    if output_dir is None:
        output_dir = "learningdata"
    
    # 画像を開く
    img = Image.open(image_path)
    width, height = img.size
    
    # 分割サイズを計算
    segment_width = width // 3
    segment_height = height // 3
    
    # 分割した画像を保存するリスト
    segments = []
    
    # 画像を分割
    for i in range(3):
        for j in range(3):
            left = j * segment_width
            top = i * segment_height
            right = left + segment_width
            bottom = top + segment_height
            
            # 最後の列/行は端まで
            if j == 2:
                right = width
            if i == 2:
                bottom = height
            
            segment = img.crop((left, top, right, bottom))
            segments.append(segment)
    
    # 一時ディレクトリを作成
    os.makedirs(temp_dir, exist_ok=True)
    
    # 一時ファイルに保存
    temp_files = []
    for idx, segment in enumerate(segments):
        temp_path = os.path.join(temp_dir, f'{idx + 1}.png')
        segment.save(temp_path)
        temp_files.append(temp_path)
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # タイムスタンプ付きでリネーム・移動
    current_datetime = datetime.datetime.now()
    output_files = []
    
    for n in range(1, 10):
        original_filename = os.path.join(temp_dir, f'{n}.png')
        new_filename = f'{current_datetime.strftime("%Y%m%d%H%M%S")}({n}).png'
        output_path = os.path.join(output_dir, new_filename)
        
        os.rename(original_filename, output_path)
        output_files.append(output_path)
    
    return output_files


def main():
    """コマンドラインエントリーポイント"""
    import argparse
    
    parser = argparse.ArgumentParser(description='画像を3x3の9パネルに分割')
    parser.add_argument('image', nargs='?', default='sample.png', 
                        help='入力画像パス (デフォルト: sample.png)')
    parser.add_argument('-o', '--output', default='learningdata',
                        help='出力ディレクトリ (デフォルト: learningdata)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    print(f"Splitting image: {args.image}")
    output_files = split_image_9panels(args.image, args.output)
    
    print(f"Generated {len(output_files)} files:")
    for path in output_files:
        print(f"  - {path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
