'''
Created on 5/05/2019

@author: Usuario
'''
import numpy as np
import pathlib
import random
import tensorflow as tf
from predict_client.prod_client import ProdClient

HOST = '192.168.0.21:9000'
# a good idea is to place this global variables in a shared file
MODEL_NAME = 'test'
MODEL_VERSION = 1

client = ProdClient(HOST, MODEL_NAME, MODEL_VERSION)

with tf.Session() as sess:
    data_root=pathlib.Path("E:\DATA\DefacedAndUndefacedWebsitesRepresentativeImages")
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    image_count=len(all_image_paths)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    
    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_images(image, [160, 160])
        image /= 255.0  # normalize to [0,1] range
        return image
    
    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        return preprocess_image(image)
    
    images=[]
    i=0
    dataNum=10
    for path in all_image_paths:
        images.append(load_and_preprocess_image(path))
        i=i+1
        if i==dataNum:
            break
    image_ds = tf.convert_to_tensor(images,tf.float32)
    indexes=tf.convert_to_tensor(all_image_labels,tf.int32)
    label_ds = tf.cast(tf.map_fn(lambda z: tf.one_hot(z,2,dtype=tf.int32),indexes),tf.float32)
    print(label_ds.eval()[0:dataNum])
    req_data = [{'in_tensor_name': 'inputs', 'in_tensor_dtype': 'DT_FLOAT', 'data': image_ds.eval()}]

prediction = client.predict(req_data, request_timeout=10)

print(prediction)