from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from flask_cors import CORS
from dotenv import load_dotenv


app = Flask(__name__)
load_dotenv()
CORS(app)

model = tf.keras.models.load_model(os.getenv('H5_FILE_PATH'))
img_height = 224
img_width = 224

@app.route('/classify', methods=['POST'])
def classify():
    fileName = request.json['filename']
    supported_extensions = ['.jpg', '.jpeg', '.png']
    
    image_path = None

    for ext in supported_extensions:
        path = os.getenv('IMAGES_PATH') + fileName + ext
        if os.path.isfile(path):
            image_path = path
            break
    
    if image_path is None:
        return jsonify({'error':'Image not found'})
    
    img = Image.open(image_path)

    img = img.resize((img_height, img_width))

    img_array = np.reshape(img,(1,img_height,img_width,3))
    pred=model.predict(img_array)
    idx=np.argmax(pred)

    return jsonify({'result': str(idx)})



if __name__ == '__main__':
    app.run()
