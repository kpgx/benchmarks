import os
import cv2
import subprocess
import time
from tqdm import tqdm

# region = "us"
# res = "100q"

in_file_dir = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us/720p/"

haar_detected_file_list = "/Users/larcuser/Projects/openalpr_benchmarks/haar_ok_file_list.txt"
haarcascade_russian_plate_number = "haarcascade_russian_plate_number.xml"

SCALE = 16
OUT_EXT = "jpg"

out_file_dir = "/Users/larcuser/Projects/openalpr_benchmarks/img_out3/us/720p_base_haar_mrim_{}_scale{}/".format(OUT_EXT, SCALE)


lic_data = cv2.CascadeClassifier(haarcascade_russian_plate_number)

def get_file_size_in_bytes(file_path):
    return os.path.getsize(file_path)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_annotated_img(img_file_path, coordinates, plate, color):
    img = cv2.imread(img_file_path)
    x1, y1 = coordinates[:2]
    x2, y2 = x1+coordinates[2], y1+coordinates[3]
    img_rect = cv2.rectangle(img, (x1,y1),(x2,y2),color, 2, cv2.LINE_4)
    img_rect_text = cv2.putText(img_rect, plate, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2, cv2.LINE_4)
    return img_rect_text


def get_haar_detected_plate_coordinates(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    numbers = lic_data.detectMultiScale(gray, 1.2)
    # for number_set in numbers:
    #     (x1, y1, w, h) = number_set
    #     x2 = x1 + w
    #     y2 = y1 + h
    return numbers


# def get_files_in_dir(a_dir, ext):
#     return sorted([name for name in os.listdir(a_dir)
#             if name.lower().endswith(ext)])


def get_mrim_img(m_img_path, plate_coordinates):
    m_img = cv2.imread(m_img_path)
    m_img_copy = m_img.copy()
    h, w, _ = m_img_copy.shape
    m_img_copy = cv2.resize(m_img_copy, (int(w / SCALE), int(h / SCALE)))
    m_img_copy = cv2.resize(m_img_copy, (w, h), interpolation=cv2.INTER_NEAREST)
    for plate_coordinate_set in plate_coordinates:
        (x, y, w, h) = plate_coordinate_set
    # x1, y1 = plate_coordinates[:2]
    # x2, y2 = x1 + plate_coordinates[2], y1 + plate_coordinates[3]
        plate_area = m_img[y:y+h, x:x+w]
        m_img_copy[y:y+h, x:x+w] = plate_area
    return m_img_copy


def write_img_to_disk(m_img, m_img_path, m_ext):
    cv2.imwrite("{}.{}".format(m_img_path, m_ext), m_img)


jpg_file_list = []
with open(haar_detected_file_list, "r") as f:  jpg_file_list = f.readlines()

make_dir(out_file_dir)

for jpg_file_name in tqdm(jpg_file_list):
    is_top_match = False
    is_in_predictions = False

    jpg_file_name = jpg_file_name.strip()
    jpg_path = os.path.join(in_file_dir, jpg_file_name.strip())
    coordinates = get_haar_detected_plate_coordinates(jpg_path)

    # print(jpg_path, coordinates)

    mrim_img = get_mrim_img(jpg_path, coordinates)

    # cv2.imshow("MRIM", mrim_img)
    # cv2.waitKey()

    out_img_path_wo_ext = os.path.join(out_file_dir, jpg_file_name.split('.')[0])
    write_img_to_disk(mrim_img, out_img_path_wo_ext, OUT_EXT)

print("that's all folks")

