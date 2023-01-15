import numpy as np
import cv2
import os

from matplotlib import pyplot as plt

# Note: some images have to be manually modified to remove
# the tape labels on the pots. GIMP was used to black them out.

# TODO: change 2 ==> 16
for i in range(1, 16):
    img = cv2.imread(os.getcwd() + '/1-11/' + f'{i}.jpg')

    # Crop out tape (green/yellow was not a good tape color choice)
    w, h, _ = img.shape
    w /= 2
    h /= 2
    around_center = 900
    img = img[int(w-around_center):int(w+around_center), int(h-around_center):int(h+around_center)]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((10, 10), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    result = img.copy()
    lower = np.array([21,78,132])
    upper = np.array([49,177,255])
    mask = cv2.inRange(img, lower, upper)

    # https://stackoverflow.com/questions/42592234/python-opencv-morphologyex-remove-specific-color
    result = cv2.bitwise_and(result, result, mask=mask)

    #cv2.imshow('original', img)
    cv2.imshow('green', result)
    # space to skip through images to see if any errors
    while True:
        if cv2.waitKey(27) == 32:
            break
        else:
            pass

    avg_color_per_row = np.average(result, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    r = round(avg_color[2], 3)
    g = round(avg_color[1], 3)
    b = round(avg_color[0], 3)
    print(f'Image {i} ==> ({r}, {g}, {b})')
