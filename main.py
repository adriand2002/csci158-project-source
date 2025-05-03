import os

import cv2

import matplotlib.pyplot as mpplot
import matplotlib.image as mpimg

from skimage.morphology import skeletonize

# Image preprocessing
def preprocess(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    # Step 1: Enhancement
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 2: Binarization
    _, img = cv2.threshold(img, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Skeletonization
    img = skeletonize(img.astype(bool)) # converts to bool (0-1)
    img = (img * 255).astype("uint8")   # reconvert back to 0-255 range

    return img

if __name__  == "__main__":
    path = "./dataset/000/L/000_L0_0.bmp"

    img = preprocess(path)

    os.environ["QT_QPA_PLATFORM"] = "xcb" # Workaround for Wayland Qt dependency issue
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()