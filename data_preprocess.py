import cv2
import numpy as np
from matplotlib import pyplot as plt

def img_preprocess(img):
    # Crop image to Height[60:135] and the whole width
    cropped_img = img[60:135, :] # [H:W]
    
    # Convert img to YUV color space
    yuv_cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2YUV)

    # Resize to 200x66 (WxH)
    resized_yuv = cv2.resize(yuv_cropped_img, (200, 66)) # (W, H)

    # Normalize pixel values in [0:255] to [0:1] with division
    resized_yuv_norm = resized_yuv / 255

    # Blur with (5,5) kernel and inferred sigma
    processed_img = cv2.GaussianBlur(resized_yuv_norm, (5,5), 0)

    return processed_img


# img = cv2.imread('test.jpg')
# processed_img = img_preprocess(img)
# cv2.imshow('Test Img', processed_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()