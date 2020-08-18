# Necessary imports
import cv2
import dlib
import numpy as np
import os



## set directories
os.chdir('PATH_TO_DIR')
path = 'IMAGE_PATH'

#Initialize color [color_type]
color_cyan = (255,200,0)
color_black = (0, 0, 0)

# Loading the image and converting it to grayscale
img= cv2.imread('image/11.jpg')
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

"""
Detecting faces in the grayscale image and creating an object - faces to store the list of bounding rectangles coordinates
The "1" in the second argument indicates that we should upsample the image 1 time.  
This will make everything bigger and allow us to detect more faces
"""

faces = detector(gray, 1)

# printing the coordinates of the bounding rectangles
print(faces)
print("Number of faces detected: ", len(faces))

"""
# Using a for loop in order to extract the specific coordinates (x1,x2,y1,y2)
for face in faces:
  x1 = face.left()
  y1 = face.top()
  x2 = face.right()
  y2 = face.bottom()
  # Drawing a rectangle around the face detected
  cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0),3)

cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
Detecting facial landmarks using facial landmark predictor dlib.shape_predictor from dlib library
This shape prediction method requires the file called "shape_predictor_68_face_landmarks.dat" to be downloaded
Source of file: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""
# Path of file
p = "shape_predictor_68_face_landmarks.dat"
# Initialize dlib's shape predictor
predictor = dlib.shape_predictor(p)

# Get the shape using the predictor

for face in faces:
    landmarks = predictor(gray, face)

    # for n in range(0,68):
    #     x = landmarks.part(n).x
    #     y = landmarks.part(n).y
    #     img_landmark = cv2.circle(img, (x, y), 4, (0, 0, 255), -1)


    points = []
    for i in range(1, 16):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)
    # print(points)

    # Ellipse parameters for high, round coverage mask
    top_ellipse = landmarks.part(27).y + (landmarks.part(28).y - landmarks.part(27).y) / 2
    centre_x = landmarks.part(28).x
    centre_y = landmarks.part(8).y - ((landmarks.part(8).y - (top_ellipse)) / 2)
    # (height of ellipse)
    axis_major = (landmarks.part(8).y - top_ellipse) / 2
    # (width of ellipse)
    axis_minor = ((landmarks.part(13).x - landmarks.part(3).x) * 0.8) / 2

    centre_x = int(round(centre_x))
    centre_y = int(round(centre_y))
    axis_major = int(round(axis_major))
    axis_minor = int(round(axis_minor))

    centre = (centre_x, centre_y)
    axes = (axis_major, axis_minor)

    # Using Python OpenCV – cv2.ellipse() method to draw mask outline for mask
    # change last parameter - line thickness and color_type for various combination
    img_2 = cv2.ellipse(img, centre, axes, 0, 0, 360, color_type, thickness=2)



    # Using Python OpenCV – cv2.ellipse() method to draw mask outline for mask
    # change last parameter - line thickness to negative for fill and color_type for various combination
    img_3 = cv2.ellipse(img, centre, axes, 0, 0, 360, color_type, thickness=-1)

# cv2.imshow("image with mask outline", img_2)
cv2.imshow("image with mask", img_3)

cv2.waitKey(0)
cv2.destroyAllWindows()
