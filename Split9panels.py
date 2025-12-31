from PIL import Image
import os
import datetime

# 画像を開く
img = Image.open('sample.png')

# 画像サイズを取得
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

        # 切り取り範囲を調整
        if j == 2:  # 最後の列
            right = width
        if i == 2:  # 最後の行
            bottom = height

        # 画像を切り取る
        segment = img.crop((left, top, right, bottom))
        segments.append(segment)

# 分割した画像をファイルに保存
for idx, segment in enumerate(segments):
    segment.save(f'temppicture/9panel/{idx+1}.png')

# 分割した画像ファイルの保存先ディレクトリ
source_directory = '.'

# 画像ファイルを移動する先のディレクトリ（'learningdata'）
destination_directory = 'learningdata'

# 現在の日時を取得
current_datetime = datetime.datetime.now()

# learningdata フォルダが存在しない場合は作成
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# 画像ファイルをリネームして移動
for n in range(1, 10):
    # 元のファイル名
    original_filename = f'temppicture/9panel/{n}.png'

    # 新しいファイル名（YYMMDDhhmmss(n)の形式）
    new_filename = f'{current_datetime.strftime("%Y%m%d%H%M%S")}({n}).png'

    # ファイルをリネームして移動
    os.rename(original_filename, f'{destination_directory}/{new_filename}')

