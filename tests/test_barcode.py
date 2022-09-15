import cv2

from tests.test_utils import image_file_path as img_path
from visual_label_compare.image_util import apply_mask
from pyzbar.pyzbar import decode


def test_barcode():
    # GIVEN
    label1a = cv2.imread(img_path("label1a.png"))
    mask = cv2.imread(img_path("label-mask.png"), cv2.IMREAD_GRAYSCALE)
    # get barcode area
    masked1a = apply_mask(label1a, mask, invert=True)

    # WHEN
    detected_barcodes = decode(masked1a)

    # THEN
    assert len(detected_barcodes) == 1
    assert str(detected_barcodes[0].data.decode("utf-8")).endswith("92055901755477000000000015")
