from __future__ import division, print_function
import sys
import os
import glob
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras import backend
from tensorflow.keras import backend
import tensorflow as tf

global graph

graph = tf.compat.v1.get_default_graph()
from skimage.transform import resize
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model('cnn.h5', compile=False)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        with graph.as_default():
            pred = model.predict_classes(x)

            index = ['NORMAL', 'PNOUMONIA']

            text = 'prediction:' + str(index[pred[0][0]])
            return text


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
