import cv2
import numpy as np

def edge_blur(image: np.ndarray, size=11):
    thresh = image[:, :, 3]
    image = image[:, :, :3]

    blurred_img = cv2.GaussianBlur(image, (size, size), 0)
    blurred_thresh = cv2.GaussianBlur(thresh, (size, size), 0)
    mask = np.zeros(image.shape, np.uint8)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(mask, contours, -1, (255, 255, 255), 5)
    blur_img = np.where(mask == np.array([255, 255, 255]), blurred_img, image)

    return np.concatenate((blur_img, blurred_thresh[:, :, None]), axis=2)