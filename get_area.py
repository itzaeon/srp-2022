import cv2
import numpy as np
import os
from rich import print, inspect

# Image path based on current directory
image_path = os.getcwd() + '/images/rotated_baby_lettuce_with_penny.png'
# Whether or not to show what's going on
debug = False

if debug: print(f'debug: {debug}')
if debug: print(f'image_path: {image_path}')

# Perform operations to remove noise from the image. 
base_img = cv2.imread(image_path)
gray_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0) # TODO: checkout this value more

# Binarize the image (blur_img --> pixels of either 0 or 255)
# with cv2.THRESH_OTSU - computes the best thresholding value
_, otsu = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Attempt to remove holes that might exist in the leaf
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

# Get the outermost contour of the leaf. Find only the outermost
# with cv2.RETR_EXTERNAL, use cv2.CHAIN_APPROX_NONE over cv2.CHAIN_APPROX_SIMPLE
# to prevent loss of points if cv2.CHAIN_APPROX_SIMPLE was used. Even though
# this uses much more memory, it *should* end up being more accurate.
# Also returns the contour of the penny or other object. A penny is used for this.
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Find both the leaf and the penny. If there are more or less contours,
# exit the program since measuring cannot continue.
if debug: print(f'contours in image: {len(contours)}')
if len(contours) != 2:
    print('error: could not detect 2 contours!')
    exit(1)

# This means that the penny **must** be smaller than the leaf,
# otherwise the script will measure the penny as the leaf and vice versa.
sorted_contours = sorted(contours, key=cv2.contourArea)

# Create blank images to draw the penny and leaf on.
img_shape = [base_img.shape[0], base_img.shape[1], 1]
drawn_penny = np.zeros(shape=img_shape, dtype=np.uint8)
drawn_leaf = np.zeros(shape=img_shape, dtype=np.uint8)

# Draw the respective contours on the blank images.
drawn_penny = cv2.drawContours(drawn_penny, sorted_contours, 0, (255, 255, 255), -1)
drawn_leaf = cv2.drawContours(drawn_leaf, sorted_contours, 1, (255, 255, 255), -1)
drawn_all = cv2.drawContours(base_img.copy(), sorted_contours, -1, (255, 0, 0), 4)

if debug:
    cv2.imshow('drawn_penny', drawn_penny)
    cv2.waitKey(0)
    cv2.imshow('drawn_leaf', drawn_leaf)
    cv2.waitKey(0)
    cv2.imshow('drawn_all', drawn_all)
    cv2.waitKey(0)

# The diameter of a penny is 19.05 millimeters, according to this source:
# https://www.usmint.gov/learn/coin-and-medal-programs/coin-specifications
# We can use this number combined with cv2.minAreaRect to get the length
# and width of the leaf and the diameter of the penny --> leaf_area_in_cm_squared.

# Using cv2.minAreaRect, get the minimum sized rectangle that will
# fit around the leaf. It returns ((x, y), (w, h), theta) where theta
# is the angle that the box is rotated at to fit around the leaf best.
leaf_rect = cv2.minAreaRect(sorted_contours[1])
leaf_length, leaf_width = (leaf_rect[1][0], leaf_rect[1][1])

# TODO: change to a circle/oval
penny_rect = cv2.minAreaRect(sorted_contours[0])
penny_diameter = penny_rect[1][0]

leaf_area_in_cm_squared = (leaf_length * 19.05 / penny_diameter) * (leaf_width * 19.05 / penny_diameter) / 100
print(f'leaf area (cm^2): {leaf_area_in_cm_squared}')

if debug:
    cv2.imshow('drawn_all', drawn_all)
    cv2.waitKey(0)
