#Author: Tarteel Alkaraan (25847208)
#Last Updated: 06/12/2024
#Import Libraries
import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt

#Facial Attribute Analysis
img_path = 'Model/Photos/1happy.png'
obj = DeepFace.analyze(img_path = img_path)
print(obj)

#Define Size Of Kernel For Gaussian Blurring
ksize = (7, 7)

#Read Image From File
image = cv2.imread('Model/Photos/1happy.png', cv2.IMREAD_GRAYSCALE)

#Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(image, ksize, sigmaX = 0, sigmaY = 0)

#Apply Sobel Edge Detection
SobelX = cv2.Sobel(src = blurred_image, ddepth = cv2.CV_64F, dx = 1, dy = 0)
SobelY = cv2.Sobel(src = blurred_image, ddepth = cv2.CV_64F, dx = 0, dy = 1) 
SobelXY = cv2.Sobel(src = blurred_image, ddepth = cv2.CV_64F, dx = 1, dy = 1)

#Display Original And Blurred Images
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image)

#Display Sobel Edge Detection
cv2.imshow('Sobel X', SobelX)
cv2.imshow('Sobel Y', SobelY)
cv2.imshow('Sobel X & Y', SobelXY)
cv2.waitKey(0)
cv2.destroyAllWindows()