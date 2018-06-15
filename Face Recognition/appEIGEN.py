import time
import os
import cv2
import numpy as np
from PIL import Image

def get_training_data(face_cascade,data_dir):
	images = []
	labels = []
	image_files=[os.path.join(data_dir,f) for f in os.listdir(data_dir)if not f.endswith('.wink')]

	for image_file in image_files:
		img = Image.open(image_file).convert('L')
		img= np.array(img,np.uint8)

		filename=os.path.split(image_file)[1]
		true_person_number = int(filename.split('.')[0].replace('subject',''))

		faces = face_cascade.detectMultiScale(img,1.05,6)
		for face in faces:
			x,y,w,h=face
			face_region = img[y:y+h,x:x+w]
			face_region=cv2.resize(face_region,(150,150))				#for Eigen and Fisher,the regions and all training data are in exactsame dimension

			images.append(face_region)
			labels.append(true_person_number)

	return images,labels

def evaluate(recognizer,face_cascade,data_dir):
	image_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir)if f.endswith('.wink')]
	num_correct=0
	
	for image_file in image_files:
		img = Image.open(image_file).convert('L')
		img= np.array(img,np.uint8)
		filename=os.path.split(image_file)[1]
		true_person_number = int(filename.split('.')[0].replace('subject',''))
		
		faces = face_cascade.detectMultiScale(img,1.05,6)
		for face in faces:
			x,y,w,h=face
			face_region = img[y:y+h,x:x+w]
			face_region=cv2.resize(face_region,(150,150))

			predicted_person_number,confidence=recognizer.predict(face_region)
			if predicted_person_number == true_person_number:
				print "Correctly Classified %d with confidence %f"%(true_person_number,confidence)
				num_correct=num_correct+1
			else:
				print "Incorrectly Classified %d as %d"%(true_person_number,predicted_person_number)
	
	accuracy = num_correct /float(len(image_files)) *100
	print "Accuracy: %0.2f%%" % accuracy



def predict(recognizer,face_cascade,img):
	predictions = []
	faces = face_cascade.detectMultiScale(img,1.05,6)
	for face in faces:
			x,y,w,h=face
			face_region = img[y:y+h,x:x+w]
			face_region=cv2.resize(face_region,(150,150))
			start=time.time()
			predicted_person_number,confidence=recognizer.predict(face_region)
			print time.time()-start
			predictions.append((predicted_person_number,confidence))
	return predictions

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.EigenFaceRecognizer_create()

print "Getting training examples"
images,labels = get_training_data(face_cascade,'yalefaces')
print "Training..............."
start=time.time()
face_recognizer.train(images,np.array(labels))
print time.time()-start
print "Finished Training!"
evaluate(face_recognizer,face_cascade,"yalefaces")


img = Image.open('face.jpg').convert('L')
img= np.array(img,np.uint8)
print predict(face_recognizer,face_cascade,img)
