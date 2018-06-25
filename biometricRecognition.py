import numpy as np 
import imageio
import cv2
import math
import matplotlib.pyplot as plt

# Converts a RGB image to gray scale using colorimetric conversion
def grayTransform(img):
	return np.dot(img, [0.299, 0.587, 0.114])

# Calculates a weigh for a class
def weigh(M, x1, x2, hist):
	return (1/M)*np.sum(hist[x1:x2])

# Calculates mean for a class
def mean(x1, x2, hist):

	a = 0 

	for i in range(x1, x2):
		a = a+(i*hist[i])

	if np.sum(hist[x1:x2]) != 0:
		return a/(np.sum(hist[x1:x2]))
	else: 
		return 0

# Calculates variance for a class
def variance(x1, x2, mean, hist):

	a = 0

	for i in range(x1, x2):
		a = a+(math.pow(i-mean, 2)*hist[i])

	if np.sum(hist[x1:x2]) != 0:
		return a/(np.sum(hist[x1:x2]))
	else: 
		return 0

# Computes the optimal thresholding using Otsu algorithm
def otsuThresholding(img):

	# Computing histogram
	hist, bin_edges = np.histogram(img, bins='auto');
	# Converts array with edges to int
	bin_edges = bin_edges.astype(int)

	# New array to histogram
	histogram = np.zeros(255)

	# Fills new histogram
	for x in range(hist.size):
		histogram[bin_edges[x]] = hist[x]

	# Converts to int
	histogram = histogram.astype(int)

	intraclassVar = np.zeros(histogram.size)

	# Intra-class variance for each intensity
	for L in range(histogram.size):
		
		# Weigh for class A, sum of frequencies from 0 to L-1
		weighA = weigh(img.shape[0]*img.shape[1], 0, L, histogram)
		# Mean for class A
		meanA = mean(0, L+1, histogram)
		# Variance for class A
		varA = variance(0, L+1, meanA, histogram)

		# Weigh for class B, sum of frequencies from L to last item
		weighB = weigh(img.shape[0]*img.shape[1], L, histogram.size, histogram)
		# Mean for class B
		meanB = mean(L, histogram.size, histogram)
		# Variance for class B
		varB = variance(L, histogram.size, meanB, histogram)
		
		# Stores intraclass variance calculated
		intraclassVar[L] = weighA*varA + weighB*varB

		# Set the min value 
		if L is 0:
			minValue = intraclassVar[L]
		elif L != 0 and minValue > intraclassVar[L]: 
			minValue = intraclassVar[L] 

	# Returns the optimal thresholding
	return list(intraclassVar).index(minValue)

# Converts a grey scale image to binary (only black and white)
def binaryTransform(img, thresholding):

	binImg = np.zeros(img.shape)

	# Sets image values according to chosen thresholding
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			if img[x,y] > thresholding: binImg[x,y] = 1

	imageio.imwrite("bin.jpg", binImg)

	return binImg

# Detects edges of the image
def edgeDetection(binImg, image):

	# Reads binary image
    binary = binImg.astype(np.uint8)
    
    # Reads the original image
    img = image.astype(np.uint8)
    
    # Finding contours
    im2, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # Only the hand edge is left
    new_contours = []
    for c in contours:
        if cv2.contourArea(c) > 100000:
            new_contours.append(c)
            
    # Drawing the contour in the original image
    cv2.drawContours(img, new_contours, -1, (0,255,0), 3)
    
    # Plots image with contours
    plt.imshow(img)
    plt.show()
    
    return img

def main():

	filename = str(input("Digite o nome do arquivo: ")).rstrip()

	# Open the original image
	image = imageio.imread(filename)

	# Converts to gray scale
	gray = grayTransform(image).astype(int)

	# Optimal thresholding
	thresholding = otsuThresholding(gray)

	# Converts to binary
	binImg = binaryTransform(gray, thresholding)

	# Edge detection
	edgeImg = edgeDetection(binImg, image)

if __name__ == "__main__":
	main()
