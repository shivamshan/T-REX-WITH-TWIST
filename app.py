from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import random
import cv2
import base64
from io import BytesIO
from PIL import Image
import io
import numpy as np


app = Flask(__name__)
socketio = SocketIO(app)


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
	# print(data)
	image=Image.open(BytesIO(base64.b64decode(data)))

	frame=np.asarray(image)
	


	
    # Process the image frame
	
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	imgencode = cv2.imencode('.png', frame)[1]

    # base64 encode
	stringData = base64.b64encode(imgencode).decode('utf-8')
	b64_src = 'data:image/jpg;base64,'
	stringData = b64_src + stringData

    # emit the frame back
	emit('response_back', stringData)

	

	




if __name__ == '__main__':
    socketio.run(app, port='5000',debug=True)