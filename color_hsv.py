import cv2
import numpy as np
import os

from rich import print, inspect
from rich.console import Console
console = Console()

image_path = os.getcwd() + '/images/test_image.png'
console.log(f'image_path: {image_path}')

base_img = cv2.imread(image_path)
gray_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, (15, 15), 0)

# Binarize the image (blur_img --> pixels of either 0 or 255)
# with v2.THRESH_OTSU - computes the best thresholding value
_, otsu = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Attempt to remove holes that might exist in the leaf
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

# Get the outermost contour of the leaf. Find only the outermost
# with cv2.RETR_EXTERNAL, use cv2.CHAIN_APPROX_NONE over cv2.CHAIN_APPROX_SIMPLE
# to prevent loss of points if cv2.CHAIN_APPROX_SIMPLE was used. Even though
# this uses much more memory, it *should* end up being more accurate.
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Create a black image to draw the contours in white over
black_img_shape = [base_img.shape[0], base_img.shape[1], 1]
black_img = np.zeros(shape=black_img_shape, dtype=np.uint8)

# Draw the found contour in white (255, 255, 255) over the completely
# black image, creating a black/white image that cv2.countNonZero uses.
bw_contours = cv2.drawContours(black_img, contours, -1, (255, 255, 255), -1)

cv2.imshow('bw_contours', bw_contours)
cv2.waitKey(0)

total_pixels = cv2.countNonZero(bw_contours)
console.log(f'total_pixels: {total_pixels}')