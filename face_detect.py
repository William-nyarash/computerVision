from matplotlib import pyplot as plt
import cv2

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
img = cv2.imread('Photos/william.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


harraCascade = cv2.CascadeClassifier('haar_face.xml')
face_rectangle = harraCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors= 2)

for (x,y,w,z) in face_rectangle:
    cv2.rectangle(img,(x,y),(x + w,y + z),(230,12,0),thickness= 3.1)

cv2.imshow('deteced image',img)    

cv2.waitKey(0)
cv2.destroyAllWindows()