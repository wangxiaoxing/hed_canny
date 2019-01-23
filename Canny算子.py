# coding=utf-8
import cv2
import numpy as np
img=cv2.imread('input/4.jpg',0)
cv2.imwrite('./article_show/Canny_source_good.jpg',img)
threshold, imgOtsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# print(threshold)
# threshold=156*2
canny = cv2.Canny(img, int(0.5 * threshold), int(threshold))
cv2.imwrite('./article_show/Canny_result_good.jpg',canny)
# cv2.waitKey(0)
cv2.destroyAllWindows()