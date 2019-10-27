####################################################################

# OpenCV implementation of the bilateral filter
# Author: Simon Ellershaw

####################################################################

import numpy as np
import cv2
import JointBilateralFilter

#####################################################################


def concatenateImages(image_1, image_2, vert):
    spacing = 20
    h1, w1 = image_1.shape[:2]
    h2, w2 = image_2.shape[:2]
    if vert:
        h_max = h1 + h2 + spacing
        w_max = max(w1, w2)
        h0 = h1 + spacing
        w0 = 0
    else:
        h_max = max(h1, h2)
        w_max = w2 + w1 + spacing
        h0 = 0
        w0 = w1 + spacing

    merge = np.zeros((h_max, w_max,  3), np.uint8)
    merge += 255
    merge[:h1, :w1, :] = image_1
    merge[h0:h_max, w0: w_max, :] = image_2
    return merge


def gridResults(inputfname, outputfname, sigma_colour, sigma_space):

    original = cv2.imread(inputfname, cv2.IMREAD_COLOR);

    if not original is None:
        h1, w1 = original.shape[:2]
        finalImage = np.zeros((0, 0, 3), np.uint8)
        for k in (sigma_space):
            vertStrip = np.zeros((0, 0, 3), np.uint8)
            for i in (sigma_colour):
                filterImg = np.zeros((h1, w1, 3), np.uint8)
                cv2.bilateralFilter(original, 50, k, i, filterImg)
                vertStrip = concatenateImages(vertStrip, filterImg, True)
            finalImage = concatenateImages(finalImage, vertStrip, False)

        cv2.imwrite(outputfname, finalImage)
    else:
        print("No image file successfully loaded.")

if __name__ == "__main__":
    inputfname = "testImages/bilteralInput.png"
    outputfname = "testImages/bilateralFilterOutput.png"
    gridOutputfname = "testImages/grid.png"

    original = cv2.imread(inputfname, cv2.IMREAD_COLOR)
    filterImg = cv2.bilateralFilter(original, 50, 60, 200)
    cv2.imwrite(outputfname, filterImg)

    sigma_colour = [2, 5, 200]
    sigma_space = [10, 60, np.inf]
    gridResults(inputfname, gridOutputfname, sigma_colour, sigma_space)
#####################################################################
