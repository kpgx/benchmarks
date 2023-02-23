import os
import cv2
import subprocess
import time
from tqdm import tqdm

# region = "us"
# res = "100q"

in_file_dir = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us/"
out_file_dir = "/Users/larcuser/Projects/openalpr_benchmarks/mrim/us/haar_mrim_png/"

SCALE = 8
OUT_EXT = "png"


def get_file_size_in_bytes(file_path):
    return os.path.getsize(file_path)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_annotated_img(img_file_path, coordinates, plate, color):
    img = cv2.imread(img_file_path)
    # print(coordinates)
    x1, y1 = coordinates[:2]
    x2, y2 = x1+coordinates[2], y1+coordinates[3]
    img_rect = cv2.rectangle(img, (x1,y1),(x2,y2),color, 2, cv2.LINE_4)
    img_rect_text = cv2.putText(img_rect, plate, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2, cv2.LINE_4)
    return img_rect_text


def get_plate_coordinates_from_txt(img_file_path):
    txt_path = img_file_path.replace('.jpg', '.txt')
    with open(txt_path, 'r') as f:
        text = f.readline()
        coordinates = [int(i) for i in text.split()[1:-1]]
        return coordinates


def get_files_in_dir(a_dir, ext):
    return sorted([name for name in os.listdir(a_dir)
            if name.lower().endswith(ext)])


def get_mrim_img(m_img_path, plate_coordinates):
    m_img = cv2.imread(m_img_path)
    m_img_copy = m_img.copy()
    h, w, _ = m_img_copy.shape
    m_img_copy = cv2.resize(m_img_copy, (int(w / SCALE), int(h / SCALE)))
    m_img_copy = cv2.resize(m_img_copy, (w, h), interpolation=cv2.INTER_NEAREST)
    x1, y1 = plate_coordinates[:2]
    x2, y2 = x1 + plate_coordinates[2], y1 + plate_coordinates[3]
    plate_area = m_img[y1:y2, x1:x2]
    m_img_copy[y1:y2, x1:x2] = plate_area
    return m_img_copy


def write_img_to_disk(m_img, m_img_path, m_ext):
    cv2.imwrite("{}.{}".format(m_img_path, m_ext), m_img)


jpg_file_list = get_files_in_dir(in_file_dir, 'jpg')

make_dir(out_file_dir)
# make_dir(predicted_annotated_dir)

# append_line(results_file, csv_header)
for jpg_file_name in tqdm(jpg_file_list):
    is_top_match = False
    is_in_predictions = False

    jpg_path = os.path.join(in_file_dir, jpg_file_name)
    coordinates = get_plate_coordinates_from_txt(jpg_path)

    print(jpg_path, coordinates)

    mrim_img = get_mrim_img(jpg_path, coordinates)

    out_img_path = os.path.join(out_file_dir, jpg_file_name)
    write_img_to_disk(mrim_img, out_img_path, OUT_EXT)

print("that's all folks")

