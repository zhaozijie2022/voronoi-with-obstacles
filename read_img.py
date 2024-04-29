import cv2
import numpy as np


def read_img(img_path="./map-downsample-origin.bmp"):
    # read in gray
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = np.array(img, dtype=np.float32)
    return img


def obstacle_dilate(img, expansion_size=3):
    kernel = np.ones((expansion_size, expansion_size), np.uint8)
    dilated_img = 255 - cv2.dilate(255 - img, kernel, iterations=1)
    cv2.imwrite("map-dilation.bmp", dilated_img)

