
import re

import numpy as np

import pandas as pd

import os

import tensorflow as tf

from flask import Flask, app, request, render_template

from tensorflow.keras import models

from tensorflow.keras.preprocessing import image

from tensorflow.python.ops.gen_array_ops import concat

from tensorflow.keras.models import load_model



model=load_model(r"crime.h5", compile=False)

app=Flask (__name__)


"""
@app.route('/')
def home():

    return render_template('home.html')"""

@app.route('/', methods=['GET','POST']) 

def home():

    if request.method == 'POST':


        image1 = request.files['image']
        file_path='crime_flask/uploads/image.jpg'
        image1.save(file_path)


        img = image.load_img(file_path, target_size=(64, 64))

        x = image.img_to_array(img)

        x = np.expand_dims (x, axis=0)

        pred= np.argmax (model.predict(x)) 
        op= ['RoadAccidents','Fighting', 'Shoplifting', 'Shooting', 'Vandalism', 'NormalVideos', 'Explosion','Burglary','Robbery','Assault','Arrest',"Stealing",'Arson','Abuse']


        result = op[pred]

        result='The predicted output is {}'.format(str(result))
        print(result)
        print(pred)

        return render_template('predict.html',name=result)
    else:
        return render_template('home.html')



if __name__ == "__main__":

    app.run(debug=True)


