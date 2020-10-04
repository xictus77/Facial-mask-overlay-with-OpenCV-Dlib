# Necessary imports
import cv2
import dlib
import numpy as np
import os
import imutils


## set directories
os.chdir('PATH_TO_DIR')
path = 'IMAGE_PATH'

#Initialize color [color_type] = (Blue, Green, Red)
color_cyan = (255,200,0)
color_black = (0, 0, 0)

# Use input () function to capture from user requirements for mask type and mask colour
choice1 = input("Please select the choice of mask color\nEnter 1 for cyan\nEnter 2 for black:\n")
choice1 = int(choice1)

if choice1 == 1:
    choice1 = color_cyan
    print('You selected mask color = cyan')
elif choice1 == 2:
    choice1 = color_black
    print('You selected mask color = black')
else:
    print("invalid selection, please select again.")
    input("Please select the choice of mask color\nEnter 1 for cyan\nEnter 2 for black :\n")


choice2 = input("Please enter choice of mask type coverage \nEnter 1 for high \nEnter 2 for medium \nEnter 3 for low :\n")
choice2 = int(choice2)

if choice2 == 1:
    # choice2 = fmask_a
    print(f'You chosen wide, high coverage mask')
elif choice2 == 2:
    # choice2 = fmask_c
    print(f'You chosen wide, medium coverage mask')
elif choice2 == 3:
    # choice2 = fmask_e
    print(f'You chosen wide, low coverage mask')
else:
    print("invalid selection, please select again.")
    input("Please enter choice of mask type coverage \nEnter 1 for high \nEnter 2 for medium \nEnter 3 for low :\n")

# print(choice2)



# Loading the image and resizing, converting it to grayscale
img= cv2.imread(path)
img = imutils.resize(img, width = 500)
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

    # Coordinates for the additional 3 points for wide, high coverage mask - in sequence
    mask_a = [((landmarks.part(42).x), (landmarks.part(15).y)),
              ((landmarks.part(27).x), (landmarks.part(27).y)),
              ((landmarks.part(39).x), (landmarks.part(1).y))]

    # Coordinates for the additional point for wide, medium coverage mask - in sequence
    mask_c = [((landmarks.part(29).x), (landmarks.part(29).y))]

    # Coordinates for the additional 5 points for wide, low coverage mask (lower nose points) - in sequence
    mask_e = [((landmarks.part(35).x), (landmarks.part(35).y)),
              ((landmarks.part(34).x), (landmarks.part(34).y)),
              ((landmarks.part(33).x), (landmarks.part(33).y)),
              ((landmarks.part(32).x), (landmarks.part(32).y)),
              ((landmarks.part(31).x), (landmarks.part(31).y))]

    fmask_a = points + mask_a
    fmask_c = points + mask_c
    fmask_e = points + mask_e

    # mask_type = {1: fmask_a, 2: fmask_c, 3: fmask_e}
    # mask_type[choice2]


    # Using Python OpenCV – cv2.polylines() method to draw mask outline for [mask_type]:
    # fmask_a = wide, high coverage mask,
    # fmask_c = wide, medium coverage mask,
    # fmask_e  = wide, low coverage mask

    fmask_a = np.array(fmask_a, dtype=np.int32)
    fmask_c = np.array(fmask_c, dtype=np.int32)
    fmask_e = np.array(fmask_e, dtype=np.int32)

    mask_type = {1: fmask_a, 2: fmask_c, 3: fmask_e}
    mask_type[choice2]


    # change parameter [mask_type] and color_type for various combination
    img2 = cv2.polylines(img, [mask_type[choice2]], True, choice1, thickness=2, lineType=cv2.LINE_8)

    # Using Python OpenCV – cv2.fillPoly() method to fill mask
    # change parameter [mask_type] and color_type for various combination
    img3 = cv2.fillPoly(img2, [mask_type[choice2]], choice1, lineType=cv2.LINE_AA)

# cv2.imshow("image with mask outline", img2)
cv2.imshow("image with mask", img3)

#Save the output file for testing
outputNameofImage = "output/imagetest.jpg"
print("Saving output image to", outputNameofImage)
cv2.imwrite(outputNameofImage, img3)


cv2.waitKey(0)
cv2.destroyAllWindows()