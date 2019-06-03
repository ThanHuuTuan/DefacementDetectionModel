'''
Created on 25/03/2019
took from https://medium.com/data-science-group-iitr/
building-a-convolutional-neural-network-in-python-with-tensorflow-d251c3ca8117
@author: Usuario
'''
import tensorflow as tf
from model.Dataset import Dataset 
import pathlib
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import time
import random

placeholder_X = tf.placeholder(tf.float32, [None,160,160,3],name="inputs")
placeholder_y = tf.placeholder(tf.float32, [None,2])
x, y = placeholder_X, placeholder_y
x_image = x
def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
    
    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # Add the biases to the results of the convolution.
        layer += biases
        
        return layer, weights
    
def new_pool_layer(input, name):
    
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        return layer
    
def new_relu_layer(input, name):
    
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)
        
        return layer
    
def new_fc_layer(input, num_inputs, num_outputs, name):
    
    with tf.variable_scope(name) as scope:

        # Create new weights and biases.
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        
        # Multiply the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases
        
        return layer

# Convolutional Layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=3, filter_size=5, num_filters=6, name ="conv1")

# Pooling Layer 1
layer_pool1 = new_pool_layer(layer_conv1, name="pool1")

# RelU layer 1
layer_relu1 = new_relu_layer(layer_pool1, name="relu1")

# Convolutional Layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_relu1, num_input_channels=6, filter_size=5, num_filters=16, name= "conv2")

# Pooling Layer 2
layer_pool2 = new_pool_layer(layer_conv2, name="pool2")

# RelU layer 2
layer_relu2 = new_relu_layer(layer_pool2, name="relu2")

# Flatten Layer
num_features = layer_relu2.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_relu2, [-1, num_features])

# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")

# RelU layer 3
layer_relu3 = new_relu_layer(layer_fc1, name="relu3")

# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=2, name="fc2")

# Use Softmax function to normalize the output
with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, dimension=1,name="prediction")
print(y_pred.name)
y_true=y
y_true_cls = tf.argmax(y_true, dimension=1)
# Use Simple meanSquaredError
with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    
# Use Adam Optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)
    
# Accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# Initialize the FileWriter
writer = tf.summary.FileWriter("Training_FileWriter/")
writer1 = tf.summary.FileWriter("Validation_FileWriter/")

# Add the cost and accuracy to summary
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

path = "../dqn" #The path to save our model to.

num_epochs = 100
batch_size=32

saver = tf.train.Saver()



with tf.Session() as sess:
    dset=Dataset()
    image_count=dset.getNumExamples()
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    # Loop over number of epochs
    for epoch in range(num_epochs):
        
        start_time = time.time()
        train_accuracy = 0
        
        for batch in range(0, int(image_count/batch_size)):
            # Run the optimizer using this batch of training data.
            dataBatch=dset.next_batch(batch_size)
            exes=[]
            wuais=[]
            for i in dataBatch:
                exes.append(i[0])
                wuais.append(i[1])
            _, acc=sess.run([optimizer,accuracy],feed_dict = {placeholder_X:np.array(exes), placeholder_y:np.array(wuais)})
            # Calculate the accuracy on the batch of training data
            train_accuracy += acc
            # Generate summary with the current batch of data and write to file
            #summ = sess.run([merged_summary])
            #writer.add_summary(summ, epoch*int(image_count/batch_size) + batch)
        
        train_accuracy /= int(image_count/batch_size)
        
        dataBatch=dset.next_batch(batch_size)
        exes=[]
        wuais=[]
        for i in dataBatch:
            exes.append(i[0])
            wuais.append(i[1])
        # Generate summary and validate the model on the entire validation set
        summ, vali_accuracy = sess.run([merged_summary, accuracy],feed_dict = {placeholder_X:np.array(exes), placeholder_y:np.array(wuais)})
        #writer1.add_summary(summ, epoch)
        

        end_time = time.time()
        
        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\tAccuracy:")
        print ("\t- Training Accuracy:\t{}".format(train_accuracy))
        print ("\t- Validation Accuracy:\t{}".format(vali_accuracy))
    saver.save(sess,path+'/model-0.1.ckpt')
    print("Saved Model")