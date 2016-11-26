import cv2

cameraSource = 1
outfile = "face_rec_saved_model.yaml" #This is the trained model which is saved for recognition
label_name_map_file = "face_rec_labels.bin" #This contains the name:label mapping

cascadePath = "bin/haarcascade_frontalface_default.xml"
cascadePath2 = "bin/haarcascade_profileface.xml"
db_path = 'images_db'
pic_format = '.png'

#Highly subject to your dataset quality, and changes for different recognizers
confidence_threshold = 100.0 #For LBPHFaceRecognizer

#Feel free to play with these parameters as you see fit
scaleFactor = 1.15
minNeighbors = 5
minSize = (30, 30)
cascadeFlags = cv2.cv.CV_HAAR_SCALE_IMAGE
face_resolution = (150, 150)