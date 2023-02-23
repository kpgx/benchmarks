import os
import cv2
import subprocess
import time
from tqdm import tqdm


jpeg_quality = 1  # max = 100
file_dir = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us"
out_file_dir = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us/{}q".format(jpeg_quality)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_img_with_(img, height, width):
    return cv2.resize(img, (width, height))


def get_files_in_dir(a_dir, ext):
    return sorted([name for name in os.listdir(a_dir)
            if name.lower().endswith(ext)])


jpg_file_list = get_files_in_dir(file_dir, 'jpg')
make_dir(out_file_dir)

for jpg_file_name in tqdm(jpg_file_list):
    jpg_path = os.path.join(file_dir, jpg_file_name)
    img = cv2.imread(jpg_path)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    new_jpg_path = os.path.join(out_file_dir, jpg_file_name)
    cv2.imwrite(new_jpg_path, img, encode_params)
