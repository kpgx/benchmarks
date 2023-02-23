import os
import cv2
import subprocess
import time
from tqdm import tqdm

file_dir = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us"
out_file_dir = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us/720p"
# openalpr_command = "docker run --rm -v /Users/larcuser/Projects/openalpr_benchmarks:/data:ro openalpr -c {} endtoend/{}/{}"
# results_file = "predictions.csv"
# csv_header = "img_name, ground_truth, is_top_prediction, is_in_prediction_list, pred1, conf1, pred2, conf2, pred3, conf3, pred4, conf4, pred5, conf5, pred6, conf6, pred7, conf7, pred8, conf8, pred9, conf9, pred10, conf10"
# ground_truth_annotated_dir = "/Users/larcuser/Projects/openalpr_benchmarks/img_out/ground_truth/"
# predicted_annotated_dir = "/Users/larcuser/Projects/openalpr_benchmarks/img_out/predicted/"
min_shape = 720


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_new_shape(img, m_min_shape):
    is_landscape = False
    height, width = img.shape[:2]
    if width >= height :
        is_landscape = True
    if is_landscape:
        new_height = m_min_shape
        new_width = new_height * width / height
    else:
        new_width = m_min_shape
        new_height = new_width * height / width
    return int(new_height), int(new_width)


def get_img_with_shape(img, height, width):
    return cv2.resize(img, (width, height))


def get_files_in_dir(a_dir, ext):
    return sorted([name for name in os.listdir(a_dir)
            if name.lower().endswith(ext)])


jpg_file_list = get_files_in_dir(file_dir, 'jpg')
make_dir(out_file_dir)

for jpg_file_name in tqdm(jpg_file_list):
    jpg_path = os.path.join(file_dir, jpg_file_name)
    img = cv2.imread(jpg_path)
    height, width = img.shape[:2]
    new_height, new_width = get_new_shape(img, min_shape)

    new_img = get_img_with_shape(img, new_height, new_width)

    new_jpg_path = os.path.join(out_file_dir, jpg_file_name)
    cv2.imwrite(new_jpg_path, new_img)
