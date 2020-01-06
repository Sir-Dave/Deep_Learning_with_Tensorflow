from __future__ import absolute_import, print_function, division

import tensorflow as tf
import tensorflow_datasets as tfds
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import math
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

#print(tf.__version__)
#tf.enable_eager_execution()

dataset, metadata = tfds.load("fashion_mnist",as_supervised=True,with_info=True)
train_dataset, test_dataset = dataset["train"], dataset["test"]
class_names = ["Tshirt/top", "Trousers", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag","Ankle boot"]

num_train_examples = metadata.splits["train"].num_examples
num_test_examples = metadata.splits["test"].num_examples

def normalize(images, labels):
    images= tf.cast(images, tf.float32)
    images/= 255
    return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

for image, label in test_dataset.take(1):
    break
image = image.numpy().reshape((28,28))

plt.figure()
plt.imshow(image, cmap= plt.cm.binary)
plt.colorbar()
plt.grid(False)
#plt.show()

plt.figure(figsize=(10,10))
i = 0
for image, label in test_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i+=1
#plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),padding="same", activation=tf.nn.relu,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(64,(3,3),padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print("Accuracy on test dataset",test_accuracy)


#Export the saved model
saved_model_dir = "save/fine_tuning"
tf.saved_model.save(model, saved_model_dir)

#convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

#save the model
with open("model.tflite","wb") as f:
    f.write(tflite_model)