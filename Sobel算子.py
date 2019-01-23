# coding=utf-8
import cv2
import numpy as np
img=cv2.imread('input/machine.jpg',0)
# cv2.imshow('img',img)
cv2.imwrite('./article_show/Sobel_source.jpg',img)
x=cv2.Sobel(img,cv2.CV_16S,1,0)
y=cv2.Sobel(img,cv2.CV_16S,0,1)
absX=cv2.convertScaleAbs(x)
absY=cv2.convertScaleAbs(y)

dst=cv2.addWeighted(absX,0.5,absY,0.5,0)
# cv2.imshow('absx',absX)
# cv2.imshow('absy',absY)
# cv2.imshow('dst',dst)
cv2.imwrite('./article_show/Sobel_result.jpg',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()