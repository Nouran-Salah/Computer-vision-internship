from transformers import AutoImageProcessor, MobileViTForSemanticSegmentation
import rasterio
import tensorflow as tf
import keras_hub as hub
# import huggingface_hub
from PIL import Image
from flask import Flask,render_template,request
import rasterio
import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2

app = Flask(__name__)

def preprocessImage(image):
    TRAIN_MEAN = np.array([
      396.46753, 494.62097, 822.32, 973.6752, 2090.11, 1964.0497, 
      1351.274, 102.73967, 141.80379, 300.74142, 35.10254, 9.753333, -80.3108
  ])

    TRAIN_STD = np.array([
      2.7006653e+02, 3.2597922e+02, 4.1812164e+02, 5.8670300e+02, 1.0559849e+03, 
      1.1914220e+03, 9.6176227e+02, 4.8804016e+01, 1.3649807e+03, 4.9603885e+02, 
      2.0184523e+01, 2.7758301e+01, 2.2802358e+05
  ])
    Image_with_ndwi=[]
    mean=[]
    with rasterio.open(image) as src:
        image=src.read(range(1,13)).transpose(1,2,0)
        green=image[:,:,2]
        swir2=image[:,:,6]
        ndwi = (green - swir2) / (swir2 + green + 1e-6)
        ndwi = np.expand_dims(ndwi, axis=-1)
        image_ndwi = np.concatenate((image, ndwi), axis=-1)
       
    H, W, C = image_ndwi.shape
    resized = np.zeros((128,128,C), dtype=np.float32)
    for c in range(C):
        resized[:,:,c] = cv2.resize(image_ndwi[:,:,c], (128,128)) 
    resized = np.expand_dims(resized, axis=0)
    normalized = (resized - TRAIN_MEAN) / (TRAIN_STD + 1e-6)
    return normalized


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

filepath="/content/segmentation_model.h5"
print(os.path.exists(filepath))  

model = tf.keras.models.load_model(
    "segmentation_model.h5",
    custom_objects={
        "dice_loss": dice_loss,
        "bce_dice_loss": bce_dice_loss
    }
)


@app.route("/",methods=['GET'])
def hello():
    return render_template("index.html")
  
@app.route("/", methods=['POST'])
def index():
    file = request.files["imagefile"]
    if file:
        # 1. Process for Model
        img_array = preprocessImage(file)
        
        # 2. Extract RGB for UI Display (Bands 4, 3, 2)
        with rasterio.open(file) as src:
            # Sentinel-2: Band 3=Red, 2=Green, 1=Blue
            # In your 1-12 range, these are usually indices 3, 2, 1
            rgb = src.read([3, 2, 1]).transpose(1, 2, 0).astype(np.float32)      
            # Simple rescaling for display (0-255)
            # You might need to adjust 'max_val' based on your data 
            max_val = 3000 
            rgb = np.clip(rgb / max_val, 0, 1)
            plt.imsave("static/original.png", rgb)

        # 3. Predict and Save Mask
        prediction = model.predict(img_array)
        mask=prediction[0, :, :, 0]
        threshold = 0.5
        binary_mask = (mask > threshold).astype(np.uint8)
        plt.imsave("static/prediction.png",binary_mask , cmap="gray")
        
        return render_template("index.html", prediction_generated=True)


if __name__ == "__main__":
    app.run(debug=True)