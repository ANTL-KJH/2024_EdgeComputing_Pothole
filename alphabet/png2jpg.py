import os
import hashlib
from PIL import Image


def file_hash(file_path):
    """파일의 해시 값을 계산하여 반환."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def convert_png_to_jpg(folder_path):
    seen_hashes = set()  # 중복 파일 해시를 저장하는 집합

    # 폴더와 하위 폴더를 탐색
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                png_path = os.path.join(root, file)
                jpg_path = os.path.splitext(png_path)[0] + '.jpg'

                # 파일 해시 계산
                png_hash = file_hash(png_path)
                if png_hash in seen_hashes:
                    # 중복 파일이면 삭제
                    os.remove(png_path)
                    print(f'Removed duplicate PNG file: {png_path}')
                    continue
                seen_hashes.add(png_hash)

                # 이미지 열기
                with Image.open(png_path) as img:
                    # PNG 이미지에서 투명도를 제거하고 배경을 흰색으로 설정
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                        background.paste(img, img.split()[-1])
                        img = background

                    # JPG로 저장
                    img.convert('RGB').save(jpg_path, 'JPEG')

                print(f'Converted {png_path} to {jpg_path}')

                # 원본 PNG 파일 삭제
                os.remove(png_path)
                print(f'Removed original PNG file: {png_path}')


# 실행
if __name__ == '__main__':
    folder_path = 'C:\\Users\\ANTL\\Desktop\\GitHub\\2024_EdgeComputing_Pothole\\alphabet\\alpha_class'
    convert_png_to_jpg(folder_path)

