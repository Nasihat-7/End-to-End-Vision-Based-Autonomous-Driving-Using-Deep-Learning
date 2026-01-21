
import socketio
import eventlet.wsgi
from flask import Flask

import base64, cv2
from io import BytesIO
from PIL import Image
import numpy as np

from tensorflow.keras.models import load_model
from preprocessing import image_normalized
model = load_model('Yexianglun_mountain_model2.h5')


max_speed = 20
steering_angle = -0.02
throttle = 0.3

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle':steering_angle.__str__(),
        'throttle':throttle.__str__()
    })


sio = socketio.Server()
app = Flask(__name__)
app = socketio.WSGIApp(sio, app)


@sio.on('connect')
def on_connect(sid, environ):
    print('connected')

@sio.on('telemetry')
def on_telemetry(sid, data):
    if data:
        print(data)
        speed = float(data['speed'])
        print(speed)

        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Image from Udacity Simulator', image)
        cv2.waitKey(1)
        print(image)

        image = image_normalized(image)
        steering_angle = float(model.predict(np.array([image])))

        throttle = 1.0-steering_angle**2-(speed/max_speed)**2
        send_control(steering_angle, throttle)

    else:
        sio.emit('manual', data={})

@sio.on('disconnect')
def on_disconnect(sid):
    print('disconnected')

#5、自动运行
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
