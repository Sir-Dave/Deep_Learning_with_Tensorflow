import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

celsius = np.array([-40, -10, 0, 8, 15, 22, 38],dtype=float)
farenheit = np.array([-40, 14, 32, 42, 56, 79, 100],dtype=float)
#for i, c in enumerate(celsius):
    #print("{} degrees celsius = {} degrees farenheit".format(c,farenheit[i]))
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss = "mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius, farenheit, epochs= 500, verbose=False)
print("Finished training this model")
plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(history.history["loss"])
#plt.show()
print(model.predict([100]))
print("There are the layer variables {}".format(l0.get_weights()))