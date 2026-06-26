import math
#Creating a single Neuron from scratch 
#inputs 
x1 = 2.0 
x2 = 3.0 
#---Neuron 1--- 
#weights
w1 = 0.5 
w2 = -0.3 
#bias 
b1 =0.1 
#weighted-sum
z1 = x1*w1 + x2*w2+ b1
#Activation function 
output1 = math.tanh(z1)

#---Neuron 2---
#weights 
w3 = 0.8
#bias
b2 = -0.2
#weighted sum 
z2 = output1 *w3 *b2
output2 = math.tanh(z2)

print(f"Neuron 1 output: {output1}")
print(f"Neuron 2 output: {output2}")
