import cv2

img1= cv2.imread('wallet2.jpg',0)	# converts to greyscale
img2=cv2.imread('example.jpg',0)


_,thresh1=cv2.threshold(img1,160,255,cv2.THRESH_BINARY_INV)
cv2.imshow('thresh1',thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()
