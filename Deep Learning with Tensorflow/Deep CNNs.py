from __future__ import absolute_import, print_function, division

import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
_URL_ = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_dir = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=_URL_, extract=True)
base_dir = os.path.join(os.path.dirname(zip_dir), "cats_and_dogs_filtered")
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
training_cats_dir = os.path.join(train_dir, "cats")
training_dogs_dir =  os.path.join(train_dir, "dogs")
validation_cats_dir = os.path.join(validation_dir, "cats")
validation_dogs_dir = os.path.join(validation_dir, "dogs")

num_cats_tr = len(os.listdir(training_cats_dir))
num_dogs_tr = len(os.listdir(training_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

'''print("Total training cat images: ",num_cats_tr)
print("Total training dog images: ",num_dogs_tr)
print("Total validation cat images: ",num_cats_val)
print("Total validation dog images: ",num_dogs_val)

print("Total training images: ",total_train)
print("Total validation images: ",total_val)'''

BATCH_SIZE = 100
IMG_SHAPE = 150   #THe resolution of our images

train_image_generator = ImageDataGenerator(rescale = 1./255)
validation_image_generator = ImageDataGenerator(rescale = 1./255)

train_data_gen = train_image_generator.flow_from_directory(shuffle=True,
                                                           directory=train_dir, batch_size=BATCH_SIZE,
                                                           target_size= (IMG_SHAPE,IMG_SHAPE),class_mode="binary")

validation_data_gen = train_image_generator.flow_from_directory(shuffle=False,
                                                           directory=validation_dir, batch_size=BATCH_SIZE,
                                                           target_size= (IMG_SHAPE,IMG_SHAPE),class_mode="binary")

sample_training_images, _ = next(train_data_gen)

def plot_images(images_arr):
    fig, axes = plt.subplots(1,5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        plt.tight_layout()
        plt.show()

#plot_images(sample_training_images[:5])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3),activation="relu", input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
#model.summary()

EPOCHS = 100
history = model.fit_generator(train_data_gen,
                              epochs=EPOCHS,
                              steps_per_epoch=int(np.ceil(total_train/float(BATCH_SIZE))),
                              validation_steps=int(np.ceil(total_val/float(BATCH_SIZE))),
                              validation_data=validation_data_gen)

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(EPOCHS)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label= "Training Accuracy")
plt.plot(epochs_range, val_acc, label= "Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label= "Training Loss")
plt.plot(epochs_range, val_loss, label= "Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.savefig("./foo.png")
plt.show()