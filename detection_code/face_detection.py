import cv2

#loading cascades
face_cascade = cv2.CascadeClassifier("../cascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("../cascades/haarcascade_eye.xml")

#definig the function that will do the detection
# gray -> image in grayscale | frame -> colored image
def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    """return -> tupples (x,y,w,h) -> x,y -> upper left corners of the rectangle
       arguements -> (grayscale image, scale (how much the size of the image must be reduced/ or size of the filters; 1.3 means
       size of image must be reduced by 1.3 times),# neighbour zones to be accepted (the no. of classifiers to be accepted to classify the
       subregion as a face/or area of interest)"""
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        """arg1 -> image on which you need to draw the rectangle, 
        arg2 -> upperLeft corner of rectangle, arg3 -> lowerRight corner of the rectangele,
        arg4 -> color in rgb, arg5-> thickness of the rectangle"""
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame

#doing face detection from webcam
#0-> internal webcam, 1-> external webcam
video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    cv2.imshow('face_detector', detect(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
