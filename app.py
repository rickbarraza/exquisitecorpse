import json
import eventlet
import time
import base64

from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit, disconnect

eventlet.monkey_patch()

import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'shhhh!'
socketio = SocketIO(app, async_mode="eventlet")

@app.route('/')
def index():
    return render_template('exquisite_corpse.html')



@socketio.on('connect', namespace='/eq')
def app_connect():
    print("CLIENT SOCKET CONNECTED")

    # SETUP MODEL AND LOAD DATA...

    socketio.emit('model_ready', { 'status':"connected",}, namespace='/eq')



@socketio.on('line_submit', namespace='/eq')
def line_submit(data):
    # IF DATA IS EMPTY, THEN SEED WITH FIRST FEW WORDS,
    # ELSE, SEED WITH THE FULL LINE...

    print("line submission from client", data)

    # GET A LINE...
    
    socketio.emit('line_append', { 'new_line':"Here is a line from the A.I."}, namespace='/eq')



@socketio.on('disconnect', namespace='/eq')
def app_diconnect():
    print("CLIENT DISCONNECTED")

if __name__ == '__main__':
    socketio.run( app, port=5002, debug=True)
