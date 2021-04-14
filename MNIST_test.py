from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation,Flatten
import numpy as np
import cv2

model=keras.models.load_model("models/tested.hdf5")

mnist = keras.datasets.mnist #G
(train_inputs,train_targets),(test_inputs,test_targets) = mnist.load_data() #G
normalised_train_inputs = train_inputs/255
normalised_test_inputs=test_inputs/255
normalised_test_inputs_1=normalised_test_inputs.reshape(-1,1,28,28)
p=model.evaluate(normalised_test_inputs,test_targets)
print("Accuracy = ",p[1])
