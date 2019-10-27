####################################################################

# OpenCV implementation of the bilateral filter
# Author: Simon Ellershaw

####################################################################

import numpy as np
import cv2

#####################################################################


def jointBilateralFilter(noFlashFname, flashFname, sigma_colour, sigma_space, sample_size, savefname):
    """Main function"""
    # read in file
    originalImg = cv2.imread(noFlashFname, cv2.IMREAD_COLOR)
    flashImg = cv2.imread(flashFname, cv2.IMREAD_COLOR)

    # check imgs loaded
    if originalImg is None:
        print(noFlashFname + " could not be loaded")
        exit()
    elif flashImg is None:
        print(flashFname + " could not be loaded")
        exit()
    elif sample_size%2 != 1:
        print("Sample size must be an odd number")  # simplification for implementation
        exit()

    # set up variables
    h_org, w_org = originalImg.shape[:2]
    filterImg = np.zeros((h_org, w_org, 3))
    border = int((sample_size - 1) / 2)

    # border added so masks can act at edges
    originalImg = addBorder(originalImg, border, sample_size)
    flashImg = addBorder(flashImg, border, sample_size)

    intensityMatrix = getIntensity(flashImg, h_org, w_org, border)
    distanceMask = setDistanceMask(border, sigma_space)  # set generic gaussian mask to reduce computation time

    # iterate over all pixels, get original img and flash intenisty 9x9 neighbourhood. Then apply filter.
    for row in range(border, border + w_org):
        if row%25 == 0:
            print(row)
        for col in range(border, border + h_org):
            nbhood = originalImg[col - border: col + border + 1, row - border: row + border + 1, :]
            flashNbhood = intensityMatrix[col - border: col + border + 1, row - border: row + border + 1]
            filterImg[col - border, row - border, :] = jointBilateralAlgorithm(nbhood, flashNbhood, distanceMask, border, sigma_colour)

    # save file
    cv2.imwrite(savefname, filterImg)


def getIntensity(img, h, w, border):
    """Returns intensity map of a color image"""
    intensityArray = np.zeros((h + 2*border, w + 2*border), dtype=float)
    for col in range(len(img)):
        for row in range(len(img[col])):
            intensityArray[col, row] = getIntensityValue(img[col, row, :])
    return intensityArray


def getIntensityValue(pixel):
    """Returns intensity value of RBG pixel as defined by Eisemann and Durand (2004)
    https://people.csail.mit.edu/fredo/PUBLI/flash/flash.pdf """
    B = float(pixel[0])
    G = float(pixel[1])
    R = float(pixel[2])
    return (R ** 2 + G ** 2 + B ** 2) / (R + G + B)


def addBorder(img, border, sample_size):
    """Border extending outer edge added to allow mask to iterate over edges"""
    h, w = img.shape[:2]
    borderImg = np.zeros((h + sample_size - 1, w + sample_size - 1, 3), dtype=float)
    borderImg[border: h + border, border: w + border] = img  # place image in centre of borderImg
    populateBorder(borderImg, border, h, w)
    return borderImg


def populateBorder(img, border, h, w):
    """Add border by extending edge values a border number of pixels"""
    # copy 4 edges
    img[border: border + h, border - 1] = img[border: border + h, border]
    img[border: border + h, border + w] = img[border: border + h, border + w - 1]
    img[border - 1, border: border + w] = img[border, border: border + w]
    img[border + h, border: border + w] = img[border + h - 1, border: border + w]
    # copy 4 corners on the diagonal
    img[border - 1, border - 1] = img[border, border]
    img[border + h, border - 1] = img[border + h - 1, border]
    img[border - 1, border + w] = img[border, border + w]
    img[border + h, border + w] = img[border + h - 1, border + w - 1]

    # call recursively till border has been filled
    if border > 1:
        border -= 1
        h += 2
        w += 2
        populateBorder(img, border, h, w)


def setDistanceMask(border, sigma):
    """Gaussian distance mask"""
    distanceMatrix = np.zeros((2*border+1, 2*border+1))
    for row in range(len(distanceMatrix)):  # iterate over all cells in mask
        for col in range(len(distanceMatrix[row])):
            dif = np.sqrt((border - row) ** 2 + (border - col) ** 2)  # pythagorean distance
            distanceMatrix[col, row] = gaussian(sigma, dif)  # apply gaussian
    return distanceMatrix


def gaussian(sigma, x):
    """Returns result of gaussian function for mu = 0"""
    return np.exp(-(x**2)/(2*(sigma**2)))/(sigma*np.sqrt(2*np.pi))


def jointBilateralAlgorithm(nbhood, flashNbhood, distanceMatrix, border, sigma_colour):
    """Applies joint bilateral algorithm to a singe pixel"""
    bilateralPixel = np.zeros(3)
    normIteration = 0
    for row in range(len(nbhood)):  # iterate over neighbourhood
        for col in range(len(nbhood[row])):
            intensity_dif = flashNbhood[row, col] - flashNbhood[border, border]  # [border, border] is central pixel
            intensity_gaussian = gaussian(sigma_colour, intensity_dif)
            bilateralPixel += distanceMatrix[col, row]*intensity_gaussian*nbhood[col, row, :]  # JBF algorithm
            normIteration += distanceMatrix[col, row]*intensity_gaussian
    return bilateralPixel/normIteration


if __name__ == "__main__":
    # Input output files
    fname_noFlash = "testImages/noflash.jpg"
    fname_flash = "testImages/noflash.jpg"
    fname_save = "testImages/jointBilateralFilterOutput.jpg"
    # Filter parameters - current parameters give a pleasing output
    sigma_colour = 1
    sigma_space = 1.5
    sample_size = 9
    # Function call
    print("Filter progress:")
    jointBilateralFilter(fname_noFlash, fname_flash, sigma_colour, sigma_space, sample_size, fname_save)
    print("Image processing complete and file outputted to " + fname_save)
