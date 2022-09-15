# from https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
# import the necessary packages
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import matplotlib.pyplot as plt
import cv2


def diff_images(imageA, imageATitle, imageB, imageBTitle):
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	(ssim, diff) = compare_ssim(grayA, grayB, full=True)
	mse = compare_mse(grayA, grayB)
	diff = (diff * 255).astype("uint8")

	# threshold the difference image, followed by finding contours to
	# obtain the regions of the two input images that differ
	thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
	# contours = imutils.grab_contours(contours)
	for c in contours:
		# compute the bounding box of the contour and then draw the
		# bounding box on both input images to represent where the two
		# images differ
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 255, 0), 2)

	display_diff(imageA, imageATitle, imageB, imageBTitle, diff, thresh, ssim, mse)
	return ssim


def compare_mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


def display_diff(imageA, imageATitle, imageB, imageBTitle, diff, thresh, ssim, mse):
	# setup the figure
	fig = plt.figure(figsize=(10, 4), dpi=300)
	plt.suptitle(f"{imageATitle} vs {imageBTitle} Diff Overlay")
	plt.title(f"MSE: {mse:.2f}, SSIM: {ssim:.2f}")

	# show first image
	ax1 = fig.add_subplot(2, 2, 1)
	ax1.title.set_text(imageATitle)
	plt.imshow(imageA, cmap=plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax2 = fig.add_subplot(2, 2, 2)
	ax2.title.set_text(imageBTitle)
	plt.imshow(imageB, cmap=plt.cm.gray)
	plt.axis("off")

	# show the diff
	ax3 = fig.add_subplot(2, 2, 3)
	ax3.title.set_text("Diff")
	plt.imshow(diff, cmap=plt.cm.gray)
	plt.axis("off")

	# show the thresh
	ax4 = fig.add_subplot(2, 2, 4)
	ax4.title.set_text("Threshold")
	plt.imshow(thresh, cmap=plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()


def apply_mask(image, mask):
	result = image.copy()
	result[mask == 0] = 0
	result[mask != 0] = image[mask != 0]
	return result
