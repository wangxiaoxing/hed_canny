import cv2
import os
import math
# 二维数组每一行每一列的最大灰度值-最小灰度值，然后取max
def max_threshold(imgPath, distance):
    img = cv2.imread(imgPath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    left_top_gray_arr = gray_img[:distance, :distance]
    height_arr = []
    for k in range(distance):
        height_arr.append([])

    # 每一行的最大横向梯度
    dx = []
    dy = []
    for i in range(distance):
        width_arr = sorted(left_top_gray_arr[i])
        # 计算横向梯度幅值
        dx.append(width_arr[distance - 1] - width_arr[0])
        # 生成纵向梯度数组
        for j in range(distance):
            height_arr[j].append(left_top_gray_arr[i][j])
    for m in range(distance):
        height_arr[m].sort()
        dy.append(height_arr[m][distance - 1] - height_arr[m][0])
    dx_max = int(sorted(dx)[distance - 1])
    dy_max = int(sorted(dy)[distance - 1])
    dxx_max=dx_max * dx_max
    dyy_max=dy_max * dy_max
    dxy_max=round(math.sqrt((dx_max*dx_max)+(dy_max*dy_max)))
    return {'dx_max': dx_max, 'dy_max': dy_max,'dxy_max':dxy_max,'imgPath':imgPath}
    # return {'dx_max': dx_max, 'dy_max': dy_max}


# 遍历hed_test_data_resize文件夹看
# dir='./hed_test_data_resize'
# imgPaths= os.listdir(dir)
# max_thresholds=[]
# for imgPath in imgPaths:
#     max_thresholds.append(max_threshold(dir+'/'+imgPath,6))
#
# max_thresholds.sort(key=lambda x:x['dxy_max'])
# dont=max_thresholds[25:]
# print(dont)
