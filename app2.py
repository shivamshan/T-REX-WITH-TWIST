from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import random
import cv2
import base64
from io import BytesIO
from PIL import Image
import io
import numpy as np
from scipy.spatial import distance as dist
import dlib
from collections import OrderedDict


app = Flask(__name__)
socketio = SocketIO(app)


def eye_aspect_ratio(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])

    ear=(A+B)/(2.0 * C)

    return ear

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 1

COUNTER = 0
TOTAL = 0
if_blinked=0

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



def shape_to_np(shape, dtype="int"):
    
    coords = np.zeros((68,2), dtype=dtype)

    
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    
    return coords











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




detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]




@app.route('/')
def index():
    return render_template('trex.html')

@app.route('/vid')
def index2():
    return render_template('vid.html')


@socketio.on('my event')
def test_message(message):
	print(message)
	

@socketio.on('catch-frame')
def catch_frame(data):
    global if_blinked

    
    image=Image.open(BytesIO(base64.b64decode(data)))

    frame=np.asarray(image)

    frame = resize_pyim(frame, width=450)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape=predictor(gray,rect)

        shape=shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)

        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        global COUNTER
        global EYE_AR_THRESH
        global EYE_AR_CONSEC_FRAMES
        global TOTAL
        
        if(ear<EYE_AR_THRESH):
            COUNTER +=1
            if_blinked=1

        else:
            if_blinked=0
            if COUNTER>=EYE_AR_CONSEC_FRAMES:
                TOTAL +=1

                COUNTER=0
                        


	
    print(if_blinked)
    emit('response_to_rex', if_blinked,broadcast=True)

	

	




if __name__ == '__main__':
    socketio.run(app, port='5000',debug=True)