#Moch. Slaviansyah Prastio 040
 
#memanggil library numpy
import numpy as np

#inisialisasi input berdasarkan jumlah batch yaitu 6 dan panjang sesuai input layer yaitu 10
inputs  = [[1.3, 0.5, 2, 0.3, 3.1, 5, 0.2, 3, -1.3, -0.5],
           [3, 0.9, 2, 0.3, 0.3, 9, 0.2, 3, -3, -0.9],
           [8, 0.3, 2, 0.3, 0, 3, 0.2, 3, -8, -0.3],
           [1.6, 1.2, 2, 2, 6.1, 2.1, 0.2, 0.2, -1.6, -1.2],
           [6, 7, 2, 0.3, 0.6, 0.7, 0.2, 3, -6, -7],
           [1.9, 1, 2, 2.2, 9.1, 0.1, 0.2, 2.2, -1.9, -1]]

#inisialisasi weight berdasarkan jumlah neuron yaitu 5 dan panjang sesuai input layer yaitu 10
weights = [[1, 0.1, 1.9, 9.5, 0.1, 1, 9.1, 5.9, -1, -0.1],
           [1.9, 0.5, 1.9, 6.7, 9.1, 5, 9.1, 7.6, -1.9, -0.5],
           [1.3, 1.1, 1.9, 6.7, 3.1, 1.1, 9.1, 7.6, -1.3, -1.1],
           [1.0, 0.5, 2, 0.1, 0.1, 5, 0.2, 1, -1.0, -0.5],
           [2.8, 1.1, 2, 0.2, 8.2, 1.1, 0.2, 2, -2.8, -1.1]]

#inisialisasi bias berdasarkan panjang neuron = 5
biases  = [7, 8, 3.5, 2, 0.5]

#kalikan input batch dengan hasil transpose weights menggunakan dot lalu ditambah bias
layer_outputs = np.dot(inputs, np.array(weights).T) + biases

#Menampilkan hasil output
print(layer_outputs)