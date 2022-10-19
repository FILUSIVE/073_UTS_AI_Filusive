#UTS AI
#Filusive Nathan Fernanda 21091397073

#Multi Neuron

#Inisialisasi Numpy
import numpy as np

#Inisialisasi Variabel
inputs = [4.8, 6.2, 1.1, 2.6, 3.1, 1.0, 2.5, 1.7, 3.3, 3.0]
weights = [
    [2.3, 4.3, 7.4, 5.3, 9.5, 3.2, 1.4, 5.0, 2.3, 4.0],
    [2.4, 5.2, 4.4, 2.0, 2.6, 7.0, 5.5, 1.5, 9.0, 4.5],
[2.0, 1.4, 4.5, 2.0, 5.0, 2.5, 4.5, 3.0, 2.5, 8.5],
[2.0, 5.2, 3.0, 1.0, 2.0, 4.0, 2.0, 4.5, 1.5, 6.5],
[4.3, 5.0, 4.2, 2.6, 3.7, 4.8, 3.0, 4.0, 3.0, 6.0],
]
biases = [9.0, 1.0, 1.5, 4.4, 1.5]

#Output
outputs = np.dot(weights, inputs) + biases

#PrintOutput
print(outputs)