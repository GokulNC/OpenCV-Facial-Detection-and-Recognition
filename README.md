## Facial Recognition System
### (Using OpenCV in Python)

A project to detect and recognize faces from live webcams by comparing with images in database efficiently, by using the Computer Vision library *[OpenCV](http://opencv.org/)*..

### Requirements
***
* Python 2.7
* OpenCV 2.4.13 (I used the wheel *opencv_python-2.4.13-cp27-cp27m-win32.whl* from http://www.lfd.uci.edu/~gohlke/pythonlibs/ )

### Usage
***

* To capture images from webcam:
```
python photo.py
#Image will be saved as name.time.png
```
* To train the model:
```
python Train_Images.py
```
* To perform facial recognition using webcam:
```
python Recognizer.py webcam
```

#### Note:

* Edit the *configs.py* file as you see fit for your datasets.
* You may put the images to train in the *images_db* folder, under the general format `person_name.anything.png`
* (Make sure each there are atleast 3 pictures of each person in different angles for better results)
* The *Train_Images.py* script will output the trained model *face_rec_saved_model.yaml*, which will be used by *Recognizer.py* for recognition.

#### References:
http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html
https://realpython.com/blog/python/face-detection-in-python-using-a-webcam/
http://hanzratech.in/2015/02/03/face-recognition-using-opencv.html

Feel free to contribute by forking and sending PR :wink:
