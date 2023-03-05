import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread(os.getcwd() + "/6.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = np.all(img != [0, 0, 0], axis=-1)

average_color_per_row = np.average(img[mask], axis=0)
average_color = np.average(average_color_per_row, axis=0)
#average_color = np.uint8(average_color)

print("Average color of the image:", average_color)

cv2.imshow('k', average_color)
while True:
    if cv2.waitKey(27) == 32:
        break
    else:
        pass
