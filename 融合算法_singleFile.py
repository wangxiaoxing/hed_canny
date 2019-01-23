import cv2
import numpy as np
from canny_otsu_minAreaRect import canny_minAreaRect
from max_threshold import max_threshold
from pred_from_ckpt2 import pred

def hed_or_canny(imgPath):
    img=cv2.imread(imgPath)
    threshold=max_threshold(imgPath,6)
    final=np.zeros(img.shape,np.uint8)
    # 阈值>16则hed
    if(threshold['dxy_max']>16):
        hed=pred([imgPath])
        hed = cv2.GaussianBlur(hed, (7, 7), 0)
        edge_points = []
        final = np.zeros(hed.shape, np.uint8)
        [rows, cols] = hed.shape
        for i in range(rows):
            for j in range(cols):
                if hed[i, j] > 127:
                    final[i, j] = 255
                    edge_points.append([j, i])
        if len(edge_points)!=0:
            cnt = np.array(edge_points)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            for point in box:
                cv2.circle(final, (point[0], point[1]), 6, (255, 0, 0))
    else:
        final=canny_minAreaRect(imgPath)

    cv2.imshow('fin',final)
    cv2.waitKey(0)


hed_or_canny('./hed_test_data_resize/2534.jpg')
# hed_or_canny('./trad_test_data_resize/2580.jpg')