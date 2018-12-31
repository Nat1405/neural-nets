#!/usr/bin/env python
from neuralnets import Network

# Neural network to translate digits 0 to 9 to a binary representation

# Initial inputs to the network
inputs = [1,0,0,0,0,0,0,0,0,0]
# Number of neurons per hidden layer; length is number of hidden layers
hidden_layers = [4]

# Create a new neural network
n = Network(inputs, hidden_layers)

n.update_weights(1, 0, [-10,10,-10,10,-10,10,-10,10,-10,10])
n.update_weights(1, 1, [-10,-10,10,10,-10,-10,10,10,-10,-10])
n.update_weights(1, 2, [-10,-10,-10,-10,10,10,10,10,-10,-10])
n.update_weights(1, 3, [-10,-10,-10,-10,-10,-10,-10,-10,10,10])

print("Outputs: " + str(n.get_outputs()) + "\n")

inputs = [0,1,0,0,0,0,0,0,0,0]
n.update_inputs(inputs)
print("Outputs: " + str(n.get_outputs()) + "\n")

inputs = [0,0,1,0,0,0,0,0,0,0]
n.update_inputs(inputs)
print("Outputs: " + str(n.get_outputs()) + "\n")

inputs = [0,0,0,1,0,0,0,0,0,0]
n.update_inputs(inputs)
print("Outputs: " + str(n.get_outputs()) + "\n")

inputs = [0,0,0,0,1,0,0,0,0,0]
n.update_inputs(inputs)
print("Outputs: " + str(n.get_outputs()) + "\n")

inputs = [0,0,0,0,0,1,0,0,0,0]
n.update_inputs(inputs)
print("Outputs: " + str(n.get_outputs()) + "\n")

inputs = [0,0,0,0,0,0,1,0,0,0]
n.update_inputs(inputs)
print("Outputs: " + str(n.get_outputs()) + "\n")

inputs = [0,0,0,0,0,0,0,1,0,0]
n.update_inputs(inputs)
print("Outputs: " + str(n.get_outputs()) + "\n")

inputs = [0,0,0,0,0,0,0,0,1,0]
n.update_inputs(inputs)
print("Outputs: " + str(n.get_outputs()) + "\n")

inputs = [0,0,0,0,0,0,0,0,0,1]
n.update_inputs(inputs)
print("Outputs: " + str(n.get_outputs()) + "\n")
