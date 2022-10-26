# UTS 2 Filusive Nathan Fernanda 
# Multi Neuron Batch Input

#Inisialisasi nummpy
import numpy as np

# Inisialisasi Variabel Input
# 6 batcth Inputs setiap batch berisi 10 
inputs =[   
            # Inputs 1
            [1.2,-4.0,2.0,-3.22,3.5,6.5,4.9,2.13,6.8,-3.77],
            # Inputs 2
            [2.5,4.3,-2.35,4.2,3.0,5.15,4.1,3.1,7.0,-1.11],
            # Inputs 3
            [3.1,-3.05,3.2,9.0,1.20,3.75,-1.23,0.15,-4.24,1.15],
            # Inputs 4
            [4.20,-5.25,1.15,-1.29,8.9,10.3,1.88,3.4,-0.17,0.23],
            # Inputs 5
            [5.0,0.48,0.33,0.34,-0.12,0.46,0.93,-0.28,3.44,1.99],
            # Inputs 6
            [6.34,-1.14,2.27,8.09,6.61,7.24,1.91,-2.22,4.78,0.75]
        ]

# Inisialisasi Variabel Weights 1
weights_1 =[
            # Neuron 1
            [-0.25,-4.1,-2.3,-9.19,4.45,1.67,2.39,-7.14,0.99,0.9],
            # Neuron 2
            [1.19,9.81,9.99,1.34,2.55,-2.82,1.56,-0.85,2.09,7.07],
            # Neuron 3
            [-2.12,7.56,-2.25,5.66,1.04,9.15,3.14,9.91,-1.55,2.89],
            # Neuron 4
            [3.2,4.44,5.35,-1.17,3.55,9.02,-1.45,2.49,-1.11,9.06],
            # Neuron 5
            [-4.78,2.18,-4.44,9.06,-1.01,2.09,1.23,-4.04,7.34,1.69]
        ] 

# Inisialisasi bias 1
# Jumlah bias pada layer 1 berisi 5
biases_1 = [1.66,2.24,3.12,4.98,5.24]

# Inisialisasi Variabel Weights 2
# Jumlah neuron sesuai dengan jumlah bias pada layer ke 2, yaitu 3
# Di setiap neuron sesuai dengan jumlah bias pada layer 1, yaitu 5
Weights_2 =[
            # Neuron 1
            [9.06,3.45,2.51,9.14,2.64],
            # Neuron 2
            [8.6,6.8,2.97,6.12,3.99],
            # Neuron3 
            [7.06,2.34,0.76,1.45,3.42]
        ] 

# Inisialisasi bias 2
# Jumlah bias pada layer 2 berisi 3
biases_2 = [0.5,4.66,9.52]

# Perhitungan output layer 1 
layer_outputs_1 = np.dot(inputs, np.array(weights_1).T) + biases_1

# Perhitungan output layer 1 
layer_outputs_2 = np.dot(layer_outputs_1,np.array(Weights_2).T)+ biases_2

# Print layer_output 2
print(layer_outputs_2)