import cv2
import os
import cv2
import subprocess
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import imutils

import numpy as np

haarcascade_russian_plate_number = "haarcascade_russian_plate_number.xml"
haarcascade_licence_plate_rus_16stages = "haarcascade_licence_plate_rus_16stages.xml"
cars = "cars.xml"

# img_path = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us/car1.jpg"
img_path = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us/car5.jpg"
img_path = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us/cfaa9dd2-a388-4e92-bb3a-ae65c28d8139.jpg"
img_path = "test_car.png"

file_dir = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us"
out_file_dir = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us/haar/cars"
haar_detected_files_dir = "/Users/larcuser/Projects/openalpr_benchmarks/endtoend/us/haar"

lic_data = cv2.CascadeClassifier(cars)


def get_haar_detected_img(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    number = lic_data.detectMultiScale(gray, 1.2)
    # number = lic_data.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    # print("number plate detected:" + str(len(number)))
    for numbers in number:
        (x, y, w, h) = numbers
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return img


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_files_in_dir(a_dir, ext):
    return sorted([name for name in os.listdir(a_dir)
            if name.lower().endswith(ext)])


jpg_file_list = get_files_in_dir(file_dir, 'jpg')
make_dir(out_file_dir)

for jpg_file_name in tqdm(jpg_file_list):
    jpg_path = os.path.join(file_dir, jpg_file_name)
    img = cv2.imread(jpg_path)

    new_img = get_haar_detected_img(img)

    new_jpg_path = os.path.join(out_file_dir, jpg_file_name)
    cv2.imwrite(new_jpg_path, new_img)


# img = cv2.imread(img_path)
# detected = get_haar_detected_img(img)
# cv2.imshow("DETECT NUMBER", detected)
# cv2.waitKey()