#Moch. Slavianysah Prastio 040

#memanggil library numpy
import numpy as np

#inisialisasi input 10 layer
inputs = [5, 4, 3, 5.4, 6, 2, 5.3, 9, 1, 8]
 
#inisialisasi weight sesuai jumlah neuron yaitu 5, dan panjang sesuai input layer yaitu 10
weights =  [[1, 0.1, 1.9, 9.5, 0.1, 1, 9.1, 5.9, -1, -0.1],
           [1.9, 0.5, 1.9, 6.7, 9.1, 5.0, 9.1, 7.6, -1.9, -0.5],
           [1.3, 1.1, 1.9, 6.7, 3.1, 1.1, 9.1, 7.6, -1.3, -1.1],
           [1, 0.5, 2, 0.1, 0.1, 5, 0.2, 1, -1.0, -0.5],
           [2.8, 1.1, 2, 0.2, 8.2, 1.1, 0.2, 2, -2.8, -1.1]]

#inisialisasi bias sesuai panjang neuron yaitu 5
biases = [4, 8, 7.1, -4.2, 5.3]

#kalikan Weight dan input menggunakan dot lalu ditambah bias
output = np.dot(weights, inputs)+biases

#tampilkan hasil output
print(output)