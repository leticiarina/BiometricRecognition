import numpy as np 
import imageio
import matplotlib.pyplot as plt

# Converts a RGB image to gray scale using colorimetric conversion
def grayTransform(img):
	return np.dot(img, [0.299, 0.587, 0.114])

def main():

	# Open the original image
	image = imageio.imread("img1.JPG")

	# Converts to gray scale
	gray = grayTransform(image)

	plt.imshow(gray, cmap=plt.get_cmap('gray'))
	plt.show()

if __name__ == "__main__":
	main()
