import os
import cv2


def read_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files


def resize(file_name, widthNew, heightNew):
    img2 = cv2.imread(os.path.join('./hed_test_data_resize448', file_name))
    img2 = cv2.resize(img2, (widthNew, heightNew))
    cv2.imwrite(os.path.join('./hed_test_data_resize224', file_name), img2)


files = read_file_name('./hed_test_data_resize448')
for file in files:
    resize(file,224,224)
