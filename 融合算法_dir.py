import cv2
import numpy as np
from canny_otsu_minAreaRect import canny_minAreaRect
from max_threshold import max_threshold
from pred_from_ckpt3 import pred
import os

# def hed_or_canny(fileName):
#     imgPath=dir+'/'+fileName
#     img=cv2.imread(imgPath)
#
#     final=np.zeros(img.shape,np.uint8)
#     hed_paths=[]
#     # 阈值>16则hed
#
#         hed=pred(imgPath)
#         edge_points=[]
#         final = np.zeros(hed.shape, np.uint8)
#         [rows, cols] = hed.shape
#         for i in range(rows):
#             for j in range(cols):
#                 if hed[i, j] > 127:
#                     final[i,j]=255
#                     edge_points.append([j,i])
#         cnt = np.array(edge_points)
#         rect = cv2.minAreaRect(cnt)
#         box = cv2.boxPoints(rect)
#         for point in box:
#             cv2.circle(final, (point[0], point[1]), 6, (255, 0, 0))
#     else:
#         final=canny_minAreaRect(imgPath)
#
#     # cv2.imshow('fin',final)
#     # cv2.waitKey(0)
#     cv2.imwrite('./merge_hed_result/merge_'+fileName,final)

# 根据图像阈值选择算法
def choose_algor(fileName):
    imgPath=dir+'/'+fileName
    threshold = max_threshold(imgPath, 6)
    if (threshold['dxy_max'] > 15):
        hed_paths.append(imgPath)
    else:
        canny_paths.append(imgPath)


# dir='./hed_test_data_resize'
# dir='./temp_data'
dir='./hed_test_data_resize'
fileNames=os.listdir(dir)
hed_paths=[]
canny_paths=[]
for fileName in fileNames:
    choose_algor(fileName)
# hed算法传的参数是所有需要用hed的文件名数组
pred(hed_paths)
# canny算法是一个文件一个文件处理
for canny_path in canny_paths:
    fileName=canny_path.split('/')[2]
    final=canny_minAreaRect(canny_path)
    cv2.imwrite('./hed_data_merge_hed_result_15_new/merge_'+fileName,final)
