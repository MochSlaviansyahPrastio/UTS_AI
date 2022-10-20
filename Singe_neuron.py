#Moch. Slaviansyah Prastio 040

# #insialisasi numpy
import numpy as np
#inisialisasi variabel
input  = [ 5 , 1 , 9 , 2.0 , 7 , 8 , 9.5 , 3 , 4 , 2 ]

#inisialisasi variabel 
weights  = [ 2 , 1.4 , 0.6 , 7.3 , -2.4 , 7.9 , 0.6 , 2.7 , -9.7 , 4.5 ]

#inisialisasi bias
bias  =  9

#perhitungan atau rumus untuk outputya
output  =  np . dot ( weights , input ) +  bias

#untuk mengeluarkan hasil dari output
print ( output )