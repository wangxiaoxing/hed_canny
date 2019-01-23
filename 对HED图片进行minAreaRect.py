import cv2
import numpy as np
import os
# hed_result为hed图片的名字，不包含路径
# 要注意hed图片是否已经高斯滤波
def minAreaRect(hed_result):
    img = cv2.imread(dir+'/'+hed_result)
    # 高斯滤波
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 遍历边缘图
    [rows, cols] = img.shape
    print(rows, cols)
    canny_points=[]
    for i in range(rows):
        for j in range(cols):
            if img[i,j]==255:
                canny_points.append([j,i])
    print(canny_points)
    # minAreaRect
    if len(canny_points)!=0:
        cnt=np.array(canny_points)
        rect = cv2.minAreaRect(cnt)
        # 获取四边形的四个顶点
        box = cv2.boxPoints(rect)
        print(box[0])
        for point in box:
            print(point[0])
            print(point[1])
            cv2.circle(img, (point[0], point[1]), 6, (255, 0, 0))
        # cv2.imshow('minAreaRect',canny)
        # cv2.waitKey()
        cv2.imwrite(filePath+'/'+'minAreaRect_'+hed_result,img)




# 多张图片批处理
# dir='./hed_test_result_sort/minAreaRect/'
# # filePaths=[dir+'gray_GaussianBlur_100',
# #            dir+'gray_GaussianBlur_120',
# #            dir + 'gray_GaussianBlur_127_best',
# #            dir + 'gray_GaussianBlur_140']
# for filePath in filePaths:
#     hed_results = os.listdir(filePath)
#     for hed_result in hed_results:
#         minAreaRect(filePath,hed_result)

# 对hed图片进行处理
# hed产生的图片路径
dir='./hed_test_result_nofusion'
# 新生成的图片存储的路径
filePath='./hed_test_result_nofusion_minArea'
hed_results=os.listdir(dir)
count=0
for hed_result in hed_results:
    if 'gray' not in hed_result:
        count=count+1
print(count)


