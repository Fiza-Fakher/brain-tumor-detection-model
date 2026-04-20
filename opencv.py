import cv2
import numpy as np

def edgeDetection(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def findTumorContour(image, sigma=0.33):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=19)
    closed = cv2.dilate(closed, None, iterations=17)

    _, mask = cv2.threshold(closed, 155, 255, cv2.THRESH_BINARY)
    final = cv2.bitwise_and(image, image, mask=mask)

    cnts = edgeDetection(mask, sigma=sigma)
    boxes = [cv2.boundingRect(c) for c in cnts]
    return boxes, cnts, final