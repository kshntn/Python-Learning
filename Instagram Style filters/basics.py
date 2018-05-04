import cv2
import numpy as np
img = cv2.imread('index.jpeg')
"""
print img.shape
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('test',gray)
cb_img = cv2.addWeighted(img,4,img.copy(),0,100) // we use zeroes other than the img.copy() to save memory space

cb_img = cv2.addWeighted(img,4,np.zeros(img.shape,dtype=img.dtype),0,100)
cv2.imshow('color',cb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

K= np.array([
	[0,-1,0],
	[-1,5,-1],
	[0,-1,0]
])
convolved = cv2.filter2D(img,-1,K)
cv2.imshow('test',img)
cv2.imshow('color',convolved)
cv2.waitKey(0)
cv2.destroyAllWindows()
