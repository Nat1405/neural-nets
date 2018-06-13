import numpy as np
import scipy.special

class Neuron(object):
    """
    """

    def __init__(self, previous_layer):
        """
        """
        self.previous_layer = previous_layer
        self.weights = np.ones(len(self.previous_layer.activations))
        self.bias=0

    def update_activation(self):
        """
        """
        self.z = np.dot(self.previous_layer.activations, self.weights) + self.bias
        # Activation is sigmoid(z) where sigmoid comes from scipy
        self.activation = scipy.special.expit(self.z)

class Input_Layer(object):
    """Input layer to network. Takes an array of activations from 0 to 1 as input.
    """
    def __init__(self, activations):
        self.activations = activations

class Hidden_Layer(object):
    """A hidden or output layer with n neurons.
    """
    def __init__(self, n, previous_layer):
        self.neurons=[]
        self.previous_layer = previous_layer
        # Create n neurons with activations from the previous layer
        for i in range(n):
            self.neurons.append(Neuron(previous_layer))
        self.update_activations()

    def update_activations(self):
        """
        """
        # Get float activations of all neurons in this layer
        self.activations = []
        for i in range(len(self.neurons)):
            self.neurons[i].update_activation()
            self.activations.append(self.neurons[i].activation)

class Network(object):
    """
    """
    def __init__(self, activations, hidden_layers):
        self.layers = []
        self.layers.append(Input_Layer(activations))
        previous_layer = self.layers[-1]
        # Create hidden layers. Each layer gets activations from
        # the previous layer
        for i in range(len(hidden_layers)):
            # Hidden_Layer(number of neurons, previous layer to get activations)
            self.layers.append(Hidden_Layer(hidden_layers[i], previous_layer))
            previous_layer = self.layers[-1]

    def update_weight(self, layer, neuron, previous_neuron, weight):
        """Update a single weight of a single neuron
        """
        self.layers[layer].neurons[neuron].weights[previous_neuron] = weight
        # Update activations of entire network
        self.update()

    def update_weights(self, layer, neuron, weights):
        """Update all weights of a single neuron
        """
        self.layers[layer].neurons[neuron].weights = weights
        self.update()

    def update(self):
        """Recalculate activations for entire network
        """
        # Update activations of entire network
        for layer in self.layers[1:]:
            layer.update_activations()

    def update_inputs(self, inputs):
        """
        """
        self.layers[0].activations=inputs
        self.update()

    def get_outputs(self):
        """Return two decimal list of outputs
        """
        copy = np.copy(self.layers[-1].activations)
        twodecimals = ["%.2f" % v for v in copy]
        return twodecimals

    def __str__(self):
        """
        """
        # Make copy of each of the hidden layer activations
        rep = ''
        for i in range(len(self.layers)):
            # convert to two decimal places
            copy = np.copy(self.layers[i].activations)
            twodecimals = ["%.2f" % v for v in copy]
            rep += "Layer {}: ".format(str(i))+str(twodecimals)+"\n"
        return rep











#
