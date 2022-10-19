#UTS AI
#Filusive Nathan Fernanda 21091397073

#Single Neuron

#Inisialisasi Numpy
import numpy as np

#Inisialisasi Variabel
inputs = [2.0, 1.5, 1.0, 3.5, 3.0, 1.5, 4.0, 2.5, 3.0, 7.2]
weights = [1.0, 3.0, 4.0, 2.5, 1.0, 1.2, 4.0, 3.2, 2.0, 2.0]
bias = 5

#Output
outputs = np.dot(weights, inputs) + bias

#PrintOutput
print(outputs)