import cv2
import os
import matplotlib.pyplot as plt

def read_image(image_path):
    if os.path.exists(image_path):
        try:
            img = cv2.imread(image_path)
            if img is not None:
                return img
            else:
                print('Image is empty')
        
        except Exception as e:
            print(f'Error occured while reading this image: {image_path}')
    else:
        print(f'This {image_path} path does not exists')

def image_to_grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def blur_img(gray_img):
    return cv2.GaussianBlur(gray_img,(7,7),0) 

def invert_color(grayscale_img):
    return cv2.bitwise_not(grayscale_img)

def binarize_img(inverted_img):
    _,binary = cv2.threshold(inverted_img,100,255,cv2.THRESH_BINARY)
    return binary

def dilate_image(binary_image):
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thre_mor = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel3)
    return thre_mor

def find_contours(dilated_image,original_img):
    contours,hierarchy = cv2.findContours(dilated_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:15]

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return original_img