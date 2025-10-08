import cv2
import numpy as np


def save(img, output_file_name):
    # ensure output directory
    out_dir = './result'
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_file_name)

    # convert to numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # sanitize and normalize to uint8 for reliable saving
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if not cv2.imwrite(out_path, img):
        raise IOError(f'Failed to write image to {out_path}')
    

def show(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def load_image():
    IMAGE_FILE_PATH = ["A1_Images/lenna.png", "A1_Images/shapes.png"]
    lenna = cv2.imread(IMAGE_FILE_PATH[0], cv2.IMREAD_GRAYSCALE)
    shapes = cv2.imread(IMAGE_FILE_PATH[1], cv2.IMREAD_GRAYSCALE)
    return lenna, shapes