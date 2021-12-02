import numpy as np
from io import BytesIO
from urllib import request

from PIL import Image


# import tensorflow as tf
# import tensorflow.lite  as tflite

import tflite_runtime.interpreter as tflite

# interpreter = tflite.Interpreter(model_path='dog_cat.tflite')
interpreter = tflite.Interpreter(model_path='cats-dogs-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img



def predict(img_url):
    img = download_image(img_url)

    ip_shape = interpreter.get_input_details()[0]['shape'][1:3]

    ip_image = prepare_image(img, ip_shape)

    img_arr = np.array(ip_image, dtype=np.float32).reshape(interpreter.\
    get_input_details()[0]['shape'])

    img_arr = img_arr / 255

    interpreter.set_tensor(input_index, img_arr)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    return float(preds)

def decode_pred(op):
    label = 'dog' if op > 0.5 else 'cat'
    return str({'label': label, 'sigmoid_val': op })

def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    res = decode_pred(pred)
    return res
