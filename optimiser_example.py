import math
import pandas 
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_data():
    data = pandas.read_csv("beans_harvest.csv", delimiter="\t")
    del data["Year"]
    return data


#features = data[["number_of_workers"]]
#labels = data[["tons_of_beans_harvested"]]


def build_model(features):
    '''Prepares the model'''
    feature = np.array(features)
    normalizer = layers.Normalization(input_shape=[1,], axis=None)
    normalizer.adapt(feature)

    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])

    return model

def prepare_optimiser(model):
    # Keep things similar to the examples in class - SGD, SSE, which is roughly MSE
    model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='mean_squared_error')


def train_model(model, features, labels, steps=1):
    print("---------------------------")
    print("Parameters before training:")
    print_model_parameters(model)

    model.fit(
        features,
        labels,
        epochs=steps,
        verbose=steps<20)

    print("---------------------------")
    print("Parameters after training:")
    print_model_parameters(model)

    
def print_model_parameters(model):
    '''Prints the model's slope and offset parameters'''
    
    # To keep things simpler for the students we're not
    # mentioning that we've used normalisation. In normal
    # linear regression this isn't required, and everything
    # is done in a single matrix multiplication, so we've
    # opted not to add confusion and will have to account
    # for the fact that normalisation has been applied

    # It's easier to calculate the slope and offset than to
    # reverse engineer the normalisation
    predictions = model.predict([0,1])
    offset = predictions[0][0]
    slope = predictions[1][0] - predictions[0][0]

    print("Line offset (intercept):" + str(offset))
    print("Line slope:" + str(slope))

