import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    sigma = sigmoid(z)
    return (1.0 - sigma) * sigma

class Network(object):
    def __init__(self, layers):
        self.layers = layers
    
    # inputValues and expectedValues must be a numpy matrix with only 1 column
    def forwardpropagate(self, inputValues, expectedValues):
        layerOutputs[0] = self.layers[0].generateNextLayerOutput(inputValues)
        remainingLayers = self.layers[1:]
        
        for i in xrange(1, len(remainingLayers)):
            lastLayerOutput = self.layerOutputs[i-1]
            currentLayer = self.layers[i]
            layerOutputs = currentLayer.generateNextLayerOutput(lastLayerOutput)
        
        return layerOutputs
    
    def cost(self, outputValues, expectedValues):
        return np.sum(0.5*np.square(outputValues - expectedValues))
    
    def backpropagate(self, layerOutputs, inputValues, expectedValues):
        weight_deltas = []
        bias_deltas = []
        
        reversed_layers = reversed(self.layers)
        reversed_outputs = reversed(layerOutputs)
        
        # Backpropagation for the final set of weights and biases
        
        weight_deltas[0] = np.zeroes(reversed_layers[0].size, reversed_layers[1].size)
        bias_deltas [0] = np.zeroes(reversed_layers[0].size, 1)
        
        d_sigmoid_matrix = d_sigmoid(np.cross(self.reversed_layers[0].weightsMatrix, reversed_outputs[1]) + self.reversed_layers.biasesMatrix)
        
        for final_layer_index in xrange(0, reversed_layers[0].size):
            d_cost = reversed_outputs[0][1][last_index]
            d_sigmoid = d_sigmoid_matrix[second_to_last_index][0]
            
            bias_delta = d_cost*d_sigmoid
            bias_deltas[0].itemset((last_index, 1), bias_delta)
            
            for second_final_layer_index in xrange(0, reversed_layers[1].size):
                neuron_output = reversed_outputs[1][second__last_index][1]

                weight_delta = neuron_output*d_sigmoid*d_cost                
                weight_deltas[0].itemset((last_index, second_to_last_index), weight_delta)
        
        
        # Backpropagation for the remaining sets of weights and biases
        
        for row in xrange(0, weight_deltas.shape[0]):
            for column in xrange(0, weight_deltas.shape[1]):
                weight_deltas[row, column] = 
        
        for layer in reversed(self.layers):
    
    
