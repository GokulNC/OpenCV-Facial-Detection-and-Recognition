import cv2, os, sys, pickle
import numpy as np
from PIL import Image
from configs import *

testImage = sys.argv[1]

faceCascade = cv2.CascadeClassifier(cascadePath)
profileCascade = cv2.CascadeClassifier(cascadePath2)

recognizer = cv2.createLBPHFaceRecognizer()
#can also use createEigenFaceRecognizer() or createFisherFaceRecognizer() or createLBPHFaceRecognizer()
#Read http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html to understand ML behind


def predictFacesFromPhoto(image): 
	#TODO: Similarly, recognize left and right faces
	img_gray = Image.open(image).convert('L')
	img = np.array(img_gray, 'uint8')
	face = faceCascade.detectMultiScale(img, scaleFactor, minNeighbors, cascadeFlags, minSize)
	for (x, y, w, h) in face:
		name_predicted, confidence = recognizer.predict(cv2.resize(img[y: y + h, x: x + w], face_resolution))
		
		print("It is predicted as "+str(name_predicted)+" with confidence"+str(confidence))
		#cv2.imshow("Scanned Face", img[y: y+h, x: x+w])
		#cv2.waitKey(1000)
####################

def predictFacesFromWebcam(label2name_map):
	
	#http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
	video_capture = cv2.VideoCapture(cameraSource)
	frame_width = video_capture.get(3)
	frame_height = video_capture.get(4)
	
	while True:
		
		#Get a frame & Convert to grayscale
		ret, frame = video_capture.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
        ##Phase 1: FACE DETECTION
		#to detect frontal face, left&right side faces
		faces = faceCascade.detectMultiScale(gray, scaleFactor, minNeighbors, cascadeFlags, minSize)
		sidefaces_left = profileCascade.detectMultiScale(gray, scaleFactor, minNeighbors, cascadeFlags, minSize)
		sidefaces_right = profileCascade.detectMultiScale(np.fliplr(gray), scaleFactor, minNeighbors, cascadeFlags, minSize)
		
        ##Phase 2: FACE RECOGNITION
		#Predict all detected faces to the list
		#The below commented lines checks if faces overlap to avoid recognizing & drawing rectangle around same face twice
		#I commented them since they seem to be unnecessary processing overhead, and works cool without them..
		for (x, y, w, h) in faces:
			#overlaps = False
			#for(xi, yi, wi, hi) in sidefaces_left:
			#	if(x>=xi+wi or xi>=x+w or y>=yi+hi or yi>=y+h): #http://www.geeksforgeeks.org/find-two-rectangles-overlap/
			#		overlaps = False
			#	else:
			#		overlaps = True
			#		break
			#if overlaps==True:
			#	print("Overlapping left and frontal face")
			#	continue
			
			#overlaps = False
			#for(Xi, yi, wi, hi) in sidefaces_right:
			#	xi = frame_width-(Xi+wi)
			#	if(x>=xi+wi or xi>=x+w or y>=yi+hi or yi>=y+h): #http://www.geeksforgeeks.org/find-two-rectangles-overlap/
			#		overlaps = False
			#	else:
			#		overlaps = True
			#		break
			#if overlaps==True:
			#	print("Overlapping right and frontal face")
			#	continue
			
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			name_predicted, confidence = recognizer.predict(cv2.resize(gray[y: y + h, x: x + w], face_resolution))
			
			if(name_predicted!=0 and confidence<confidence_threshold):
				print("It is predicted as "+label2name_map[name_predicted]+" with confidence "+str(confidence))
				cv2.putText(frame, label2name_map[name_predicted]+":"+str(confidence), (x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
		
		
		for (x, y, w, h) in sidefaces_left:
			#overlaps = False
			#for(xi, yi, wi, hi) in faces:
			#	if(x>=xi+wi or xi>=x+w or y>=yi+hi or yi>=y+h): #http://www.geeksforgeeks.org/find-two-rectangles-overlap/
			#		overlaps = False
			#	else:
			#		overlaps = True
			#		break
			#if overlaps==True:
			#	print("Conflicting front face with left face")
			#	continue
			
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			name_predicted, confidence = recognizer.predict(cv2.resize(gray[y: y + h, x: x + w], face_resolution))
			
			if(name_predicted!=0 and confidence<confidence_threshold):
				print("LSF is predicted as "+label2name_map[name_predicted]+" with confidence "+str(confidence))
				cv2.putText(frame, label2name_map[name_predicted]+":"+str(confidence), (x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
		
		for (X, y, w, h) in sidefaces_right:
			x = int(frame_width-(X+w))
			#overlaps = False
			#for(xi, yi, wi, hi) in faces:
			#	if(x>=xi+wi or xi>=x+w or y>=yi+hi or yi>=y+h): #http://www.geeksforgeeks.org/find-two-rectangles-overlap/
			#		overlaps = False
			#	else:
			#		overlaps = True
			#		break
			#if overlaps==True:
			#	print("Conflicting front face with right face")
			#	continue
			
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			name_predicted, confidence = recognizer.predict(cv2.resize(gray[y: y + h, x: x+w], face_resolution))
			
			if(name_predicted!=0 and confidence<confidence_threshold):
				print("RSF is predicted as "+label2name_map[name_predicted]+" with confidence "+str(confidence))
				cv2.putText(frame, label2name_map[name_predicted]+":"+str(confidence), (x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
		
		
		
		cv2.imshow('Video', frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			print("\nQuitting")
			break

	video_capture.release()
	cv2.destroyAllWindows()


####_MAIN_####


if os.path.isfile(outfile): recognizer.load(outfile)
else: 
	print "Train your images first"
	exit()

with open(label_name_map_file, 'rb') as handle:
	label_name_map = pickle.load(handle)

print("Press 'q' to quit\n\n\n")
if(testImage == "webcam"):
	predictFacesFromWebcam(label_name_map)
else:
	predictFacesFromPhoto(testImage)