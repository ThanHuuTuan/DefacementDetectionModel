'''
Created on 5/05/2019

@author: Usuario
'''
import tensorflow as tf
import random
import pathlib
import numpy as np

class Dataset:

    def get(self):
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
        dataNum=10000
        for path in all_image_paths:
            images.append(load_and_preprocess_image(path))
            i=i+1
            if i==dataNum:
                break
        image_ds = tf.convert_to_tensor(images,tf.float32)
        indexes=tf.convert_to_tensor(all_image_labels,tf.int32)
        label_ds = tf.cast(tf.map_fn(lambda z: tf.one_hot(z,2,dtype=tf.int32),indexes),tf.float32)
        return image_ds.eval(),label_ds.eval()[0:dataNum]
    
    def __init__(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        x,y=self.get()
        self._data = np.array(list(zip(x,y)))
        self._num_examples = self._data.shape[0]
        pass


    @property
    def data(self):
        return self._data
    
    def getNumExamples(self):
        return self._num_examples
    
    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
    
        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
    
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]

