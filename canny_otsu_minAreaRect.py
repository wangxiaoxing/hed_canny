import cv2
import numpy as np
import os
def canny_minAreaRect(imgPath):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.medianBlur(img, 5)
    img=cv2.GaussianBlur(img,(7,7),0)
    # otsu算法获取最佳阈值
    threshold, imgOtsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny = cv2.Canny(img, int(0.5 * threshold), int(threshold))
    # 遍历边缘图
    [rows, cols] = canny.shape
    # print(rows, cols)
    canny_points=[]
    for i in range(rows):
        for j in range(cols):
            if canny[i,j]==255:
                canny_points.append([j,i])
    if len(canny_points) != 0:
        cnt=np.array(canny_points)
        rect = cv2.minAreaRect(cnt)
        # 获取四边形的四个顶点
        box = cv2.boxPoints(rect)
        for point in box:
            cv2.circle(canny, (point[0], point[1]), 6, (255, 0, 0))
        # cv2.imshow('minAreaRect',canny)
        # cv2.waitKey()
        return canny
    else:
        return canny

    # cv2.imwrite(imgPath + '_fin.jpg', canny)

def read_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files


# 批处理
# filenames = read_file_name('./canny_hed_test_result')
# # index = 0
# for filename in filenames:
#     # index = index + 1
#     # trad_edge_detection('./trad_test_input/' + filename, index)
#     canny_minAreaRect('./canny_hed_test_result/' + filename)

# test_image = './trad_test_data_resize/2580.jpg'
# canny_minAreaRect(test_image)

