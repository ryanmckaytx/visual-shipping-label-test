def apply_mask(image, mask):
	result = image.copy()
	result[mask == 0] = 0
	result[mask != 0] = image[mask != 0]
	return result
