def apply_mask(image, mask, invert=False):
	result = image.copy()
	result[(mask == 0) != invert] = 0
	result[(mask != 0) != invert] = image[(mask != 0) != invert]
	return result
