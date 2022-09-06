import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('pic.JPG')

#this method works only on grey images, so we have to convert our RGB testing image to grey skin
grey_conv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detecting the faces in the picture 
detect_faces = face_cascade.detectMultiScale(grey_conv, 1.022, 4)

#drawing rectangles arounf the image's face for visualization
for (x,y,w,h) in detect_faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)

#for image visualization
cv2.imshow('img', image)
cv2.waitKey()
