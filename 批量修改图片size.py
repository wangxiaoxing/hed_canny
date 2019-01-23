import os
import cv2


def read_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files


def resize(file_name, widthNew, heightNew):
    img2 = cv2.imread(os.path.join('./trad_test_data', file_name))
    img2 = cv2.resize(img2, (widthNew, heightNew))
    cv2.imwrite(os.path.join('./trad_test_data_resize', file_name), img2)


files = read_file_name('./trad_test_data')
for file in files:
    resize(file,448,448)
