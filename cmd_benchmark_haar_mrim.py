import os
import cv2
import subprocess
import time
from tqdm import tqdm


file_dir = "img_out3/us/720p_base_haar_mrim_jpg_scale16"
openalpr_command = "docker run --rm -v /Users/larcuser/Projects/openalpr_benchmarks:/data:ro openalpr -c us %s/{}"%file_dir
results_file = "predictions_{}.csv".format(file_dir.split("/")[-1])
csv_header = "img_name, file_size(bytes), ground_truth, is_top_prediction, is_in_prediction_list, pred1, conf1, pred2, conf2, pred3, conf3, pred4, conf4, pred5, conf5, pred6, conf6, pred7, conf7, pred8, conf8, pred9, conf9, pred10, conf10"
ground_truth_annotated_dir = "{}_ground_truth/".format(file_dir)
predicted_annotated_dir = "{}_predicted/".format(file_dir)


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


def get_plate_coordinates_from_txt(txt_path):
    # txt_path = img_file_path.replace('.jpg', '.txt')
    with open(txt_path, 'r') as f:
        text = f.readline()
        coordinates = [int(i) for i in text.split()[1:-1]]
        return coordinates


def get_real_number_plate_from_txt(txt_path):
    # txt_path = img_file_path.replace('.jpg', '.txt')
    with open(txt_path, 'r') as f:
        text = f.readline()
        plate_number = text.split()[-1]
        return plate_number


def get_files_in_dir(a_dir, ext):
    return sorted([name for name in os.listdir(a_dir)
            if name.lower().endswith(ext)])


def get_processed_output(command_output):
    str_output = command_output.decode('utf-8')
    str_output_lines = str_output.split('\n')
    result_list = []
    for line in str_output_lines[1:]:
        split_line = line.split('\t')
        if len(split_line) != 2:
            break
        plate = split_line[0].replace('-', '').strip()
        confidence = split_line[1].replace('confidence:', '').strip()
        result_list.append((plate, confidence))
        # print(plate, confidence)
    return result_list


def get_the_predicted_plate(img_path):
    formatted_command = openalpr_command.format(img_path)
    # print(formatted_command)
    result = subprocess.run(formatted_command.split(), stdout=subprocess.PIPE)
    output = result.stdout
    procceses_output = get_processed_output(output)
    return procceses_output


def append_line(file_name, txt):
    with open(file_name, 'a+') as f:
        f.write(txt+'\n')


jpg_file_list = get_files_in_dir(file_dir, 'jpg')

make_dir(ground_truth_annotated_dir)
make_dir(predicted_annotated_dir)

append_line(results_file, csv_header)
for jpg_file_name in tqdm(jpg_file_list):
    is_top_match = False
    is_in_predictions = False

    jpg_file_name = jpg_file_name.strip()
    jpg_path = os.path.join(file_dir, jpg_file_name)
    file_size = get_file_size_in_bytes(jpg_path)

    coor_path = "endtoend/us/{}.txt".format(jpg_file_name.split('.')[0])
    truth_plate_number = get_real_number_plate_from_txt(coor_path)
    coordinates = get_plate_coordinates_from_txt(coor_path)

    predicted_plates = get_the_predicted_plate(jpg_file_name)

    if len(predicted_plates):
        top_prediction = "{} {}".format(predicted_plates[0][0], predicted_plates[0][1])

        ground_truth_annotated_img = get_annotated_img(jpg_path, coordinates, truth_plate_number, (0, 255, 0))
        cv2.imwrite(os.path.join(ground_truth_annotated_dir, jpg_file_name), ground_truth_annotated_img)
        cv2.imshow("ground truth", ground_truth_annotated_img)

        predicted_annotated_img = get_annotated_img(jpg_path, coordinates, top_prediction, (0, 0, 255))
        cv2.imwrite(os.path.join(predicted_annotated_dir, jpg_file_name), predicted_annotated_img)
        cv2.imshow("predicted plate", predicted_annotated_img)

        if truth_plate_number == predicted_plates[0][0]:
            is_top_match = True
        if truth_plate_number in [i[0] for i in predicted_plates]:
            is_in_predictions = True

    all_predictions = ", ".join(["{}, {}".format(i[0], i[1]) for i in predicted_plates])
    result_line = "{}, {}, {}, {}, {}, {}".format(jpg_file_name, file_size, truth_plate_number, is_top_match, is_in_predictions, all_predictions)
    append_line(results_file, result_line)
    time.sleep(2)

print("that's all folks")

