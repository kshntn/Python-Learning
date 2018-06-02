import cv2

img1= cv2.imread('wallet2.jpg')
gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)


_,thresh1=cv2.threshold(gray,140,255,cv2.THRESH_BINARY_INV)
#thresh1=cv2.GaussianBlur(thresh1,(7,7),0) # this will create a blur so we can smooth out any small image noise and reduce the unwanted contours inside

contours,_=cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print len(contours)
contour =contours[len(contours)-1]

epsilon=0.1*cv2.arcLength(contour,True)
approx=cv2.approxPolyDP(contour,epsilon,True)

cv2.drawContours(img1,[approx],-1,(0,255,0),2)
cv2.imshow('img1',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
