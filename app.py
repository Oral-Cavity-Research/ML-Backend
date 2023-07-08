from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model('Densenet201(Original Images).h5')
img_height = 224
img_width = 224

@app.route('/classify', methods=['POST'])
def classify():
    image = request.files['image']

    img = Image.open(io.BytesIO(image.read()))

    img = img.resize((img_height, img_width))

    img_array = np.reshape(img,(1,img_height,img_width,3))
    pred=model.predict(img_array)
    idx=np.argmax(pred)

    response = jsonify({'result': str(idx)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run()
