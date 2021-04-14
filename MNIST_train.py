

from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation,Flatten
import numpy as np
import cv2

mnist = keras.datasets.mnist #G
(train_inputs,train_targets),(test_inputs,test_targets) = mnist.load_data() #G
normalised_train_inputs = train_inputs/255
normalised_test_inputs=test_inputs/255

model=keras.Sequential()
model.add(Flatten(input_shape=normalised_train_inputs.shape[1:]))
model.add(Dense(55))
model.add(Activation("relu"))
model.add(Dense(46))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.Adam(learning_rate=0.05),metrics=['accuracy'])
model.fit(normalised_train_inputs,train_targets,batch_size=10000,epochs=50)
model.save("models/tested.hdf5")
