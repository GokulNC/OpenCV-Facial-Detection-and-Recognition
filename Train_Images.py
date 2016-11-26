import cv2, os, sys, pickle
import numpy as np
from PIL import Image
from configs import *

faceCascade = cv2.CascadeClassifier(cascadePath)
profileCascade = cv2.CascadeClassifier(cascadePath2)

recognizer = cv2.createLBPHFaceRecognizer()
#can also use createEigenFaceRecognizer() or createFisherFaceRecognizer() or createLBPHFaceRecognizer()
#Read http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html to understand ML behind


def getFacesAndNames(path):
	image_paths = [os.path.join(path, f) for f in os.listdir(path)  if f.endswith(pic_format)]
	faces = []
	names = []
	count = 0
	label2name_map = {}
	name2label_map = {}
	
	for i in image_paths:
		
		#Convert to grayscale and get as np_array
		img_gray = Image.open(i).convert('L')
		width, height = img_gray.size
		img = np.array(img_gray, 'uint8')
		
		#Create label for person
		#name = int(os.path.split(i)[1].split(".")[0].replace("subject", ""))
		person_name = os.path.split(i)[1].split(".")[0]
		if name2label_map.has_key(person_name):
			name = name2label_map[person_name]
		else:
			count += 1
			name2label_map[person_name] = count
			name = count
			label2name_map[count] = person_name
		
		#to detect frontal face
		face = faceCascade.detectMultiScale(img, scaleFactor, minNeighbors, cascadeFlags, minSize)
		#to detect left side face
		sideface_left = profileCascade.detectMultiScale(img, scaleFactor, minNeighbors, cascadeFlags, minSize)
		#to detect right side face (mirror flip the image and use same cascade)
		sideface_right = profileCascade.detectMultiScale(np.fliplr(img), scaleFactor, minNeighbors, cascadeFlags, minSize)
		
		#Add all detected faces to the list
		for(x, y, w, h) in face:
			faces.append(cv2.resize(img[y: y + h, x: x + w], face_resolution))
			names.append(name)
			#cv2.imshow("Adding faces to traning set...", img[y: y + h, x: x + w])
			#cv2.waitKey(0)
			print("Frontal Face found in "+i)
		
		for(x, y, w, h) in sideface_left:
			faces.append(cv2.resize(img[y: y + h, x: x + w], face_resolution))
			names.append(name)
			#cv2.imshow("Adding faces to traning set...", img[y: y + h, x: x + w])
			#cv2.waitKey(0)
			print("Left Side Face found in "+i)
			
		for(X, y, w, h) in sideface_right:
			x = width-(X+w) #reflip to unmirror
			faces.append(cv2.resize(img[y: y + h, x: x+w], face_resolution))
			names.append(name)
			#cv2.imshow("Adding faces to traning set...", img[y: y + h, x: x + w])
			#cv2.waitKey(0)
			print("Right Side Face found in "+i)
		
		
	return faces, names, label2name_map


####_MAIN_####

faces, names, label_name_map = getFacesAndNames(db_path) #Setup the facial pictures
#cv2.destroyAllWindows()

recognizer.train(faces, np.array(names)) #Train for facial recognition
recognizer.save(outfile) #Dump the trained model
with open(label_name_map_file, 'wb') as handle:
	pickle.dump(label_name_map, handle, protocol=pickle.HIGHEST_PROTOCOL) #Dump the label:name map