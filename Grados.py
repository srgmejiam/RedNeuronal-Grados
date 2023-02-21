import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Preparar los datos
fahrenheit_q = np.array([-40, -32, -20, 0, 32, 100], dtype=float)
celsius_a = np.array([-40, -35.5556, -28.8889, 17.7778, 0, 37.7778], dtype=float)

# Definir el modelo
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=100, input_shape=[1])
])

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.098), loss='mean_squared_error')

# Entrenar el modelo
epochs_hist = model.fit(fahrenheit_q, celsius_a, epochs=1500)



plt.xlabel('Número de Epocas')
plt.ylabel("Magnitud")
plt.plot(epochs_hist.history['loss'])
plt.show()


# Evaluar el modelo
print("Modelo después de entrenar: ")
print(model.predict([100.0]))

# Usar el modelo
prediction = model.predict([100.0])
print("Predicción en grados Celsius: {}".format(prediction[0][0]))


