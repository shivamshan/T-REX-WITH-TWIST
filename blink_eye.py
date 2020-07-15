import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
import dlib
from collections import OrderedDict

def eye_aspect_ratio(eye):
	A=dist.euclidean(eye[1],eye[5])
	B=dist.euclidean(eye[2],eye[4])
	C=dist.euclidean(eye[0],eye[3])

	ear=(A+B)/(2.0 * C)

	return ear

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

def resize_pyim(image, width=None, height=None, inter=cv2.INTER_AREA):
    
    dim = None
    (h, w) = image.shape[:2]

    
    if width is None and height is None:
        return image

    
    if width is None:
        
        r = height / float(h)
        dim = (int(w * r), height)

    
    else:
        
        r = width / float(w)
        dim = (width, int(h * r))

    
    resized = cv2.resize(image, dim, interpolation=inter)

    
    return resized

def shape_to_np(shape, dtype="int"):
	
	coords = np.zeros((68,2), dtype=dtype)

	
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	
	return coords







detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]


cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)


while True:
	ret, frame = cap.read()
	frame = resize_pyim(frame, width=450)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Second parameter is the number of image pyramid layers
	rects = detector(gray, 0)

	for rect in rects:
		shape=predictor(gray,rect)

		shape=shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]

		leftEAR = eye_aspect_ratio(leftEye)

		rightEAR = eye_aspect_ratio(rightEye)

		ear = (leftEAR + rightEAR) / 2.0



		leftEyeHull=cv2.convexHull(leftEye)
		rightEyeHull=cv2.convexHull(rightEye)

		cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)

		cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)

		if(ear<EYE_AR_THRESH):
			COUNTER +=1

		else:
			if COUNTER>=EYE_AR_CONSEC_FRAMES:
				TOTAL +=1

				COUNTER=0

		cv2.putText(frame,"Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Blink Detection",frame)

	key=cv2.waitKey(1) & 0xFF

	if key==ord("q"):
		break


cap.release()
cv2.destroyAllWindows()



