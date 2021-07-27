# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:27:13 2021

@author: Owner
"""


import cv2, sys
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from matplotlib import pyplot as plt
import matplotlib.pyplot as pp
from keras.utils import np_utils
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy

imageFile='C:/Users/Owner/.spyder-py3/test2/test2.jpg'
img = cv2.imread(imageFile, cv2.IMREAD_COLOR)



blur = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
edged = cv2.Canny(blur, 10, 250)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)



contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

contours_xy = np.array(contours)
contours_xy.shape

x_min, x_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)
print(x_min)
print(x_max)
 
# y의 min과 max 찾기
y_min, y_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)
print(y_min)
print(y_max)



# image trim 하기
x = x_min
y = y_min
w = x_max-x_min
h = y_max-y_min
img_trim = img[y:y+h, x:x+w]
cv2.imwrite('org_trim.jpg', img_trim)
org_image = cv2.imread('org_trim.jpg')






gray = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)


ret, wafer = cv2.threshold(gray, 250, 10, cv2.THRESH_BINARY_INV)


ret, defect = cv2.threshold(gray, 10, 50, cv2.THRESH_BINARY_INV)

ret, defect2 = cv2.threshold(defect, 1,1, cv2.THRESH_BINARY)


kk= defect2 + wafer


image_w = 64
image_h = 64

resized = cv2.resize(kk, (64, 64))

reshaped = np.reshape(resized, (64, 64))

plt.imshow(reshaped, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()
plt.imshow(reshaped)
print(reshaped)

print(reshaped.shape)

a = [][]
for i in range(0,63):
    a = reshaped


