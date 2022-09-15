import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#creating a video capture object (to capture all frames in the video)
#0 means the prgram will be detecting video from my webcam
#you can replace 0 by 'name_of_your_video_file.mp4' to detect faces from a saved video
captr = cv2.VideoCapture(0)

#infinite loop to get each frame from the video
#captr.read() returns 2 values: a flag that indicates whether or not a frame has been read correctly, and the frame itself
while True:
    _, image = captr.read()

    #this method works only on grey images, so we have to convert our RGB testing image to grey skin
    grey_conv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #detecting the faces in the picture 
    detect_faces = face_cascade.detectMultiScale(grey_conv, 1.1, 4)

    #drawing rectangles arounf the image's face for visualization
    for (x,y,w,h) in detect_faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)

    #for image visualization
    cv2.imshow('img', image)

    #since this is an infinite loop, we need to break from it when we press on the escape key
    ok = cv2.waitKey(30) & 0xff
    if ok==27:
        break

#releasing the video capture object
captr.release()   