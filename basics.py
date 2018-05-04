import cv2
img = cv2.imread('index.jpeg')
print img.shape
cv2.imshow('test',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
