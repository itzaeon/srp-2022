import numpy as np
import cv2
import os

image = cv2.imread(os.getcwd() + '/images/baby_lettuce_with_penny.png')

channels = cv2.mean(image_bgr)

observation = np.array([(channels[2], channels[1], channels[0])])

print(observation)
