import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time


celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees celsius = {} degree fahrenheit".format(c, fahrenheit_a[i]))


model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print('Finished training the model')

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])

time.sleep(5)
print('Predicting the fahrenheit')
print(model.predict([100.0]))