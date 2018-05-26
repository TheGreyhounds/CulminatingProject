import numpy as np
import os

def sigmoid(matrix):
        return (np.zeros(matrix.size).fill(1.0) + exp(-1.0*matrix))**(-1)
        
def d_sigmoid(matrix):
    sigma = sigmoid(matrix)
    return (np.zeros(sigma.size).fill(1.0) - sigma) * sigma 


class Network(object):

    # layers must be an array of Layer objects
    def __init__(self, layers):
        self.layers = layers
        self.save_directory = "Network Data"
    
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
        # Because we are backpropagating, "previous layer" refers to the
        # layer adjacent to the current layer, on the output layer side
        
        reversed_weight_deltas = []
        reversed_bias_deltas = []
        
        reversed_partial_neuron_derivatives = []
        reversed_layers = reversed(self.layers)
        reversed_outputs = reversed(layerOutputs)
        
        for layer_number in xrange(0, len(reversed_layers)):
            reversed_weight_deltas[layer_number] = np.zeros(current_layer.weightsMatrix.size)
            reversed_bias_deltas[layer_number] = np.zeros(current_layer.biasesMatrix.size)
            
            current_layer = reversed_layers[layer_number]
            previous_layer = reversed_layers[layer_number - 1]
            previous_layer_size = previous_layer.size
            current_layer_size = current_layer.size
            reversed_partial_neuron_derivatives.append(np.zeros(current_layer_size, 1))
            
            d_sigmoid_matrix = d_sigmoid(np.dot(current_layer.weightsMatrix, reversed_outputs[layer_number+1]) + current_layer.biasesMatrix)
            
            for current_layer_neuron_number in xrange(0, current_layer_size):
                partial_derivative = 0.0
                for previous_layer_neuron_number in xrange(0, previous_layer_size):
                    weight = reversed_layers[layer_number - 1].getWeight(previous_layer_neuron_number, current_layer_neuron_number)
                    d_sigmoid = d_sigmoid_matrix[previous_layer_neuron_number, 0]
                    if layer_number is 0:
                        neuron_value = layerOutputs[currenta_layer_neuron_number, 0]
                        expected_value = expectedValues[current_layer_neuron_number]
                    else:
                        previous_partial_derivative = reversed_partial_neuron_derivatives[layer_number - 1][previous_layer_neuron_number][0]
                    partial_derivative += weight * d_sigmoid * previous_partial_derivative
                reversed_partial_neuron_derivatives[layer_number][current_layer_neuron_number][0] = partial_derivative
                
                for previous_layer_neuron_number in xrange(0, previous_layer_size):
                    previous_layer_neuron_value = previous_layer[previous_layer_neuron_number][0]
                    d_sigmoid_value = d_sigmoid_matrix[previous_layer_neuron_number][0]
                    
                    weight_delta = previous_layer_neuron_value * d_sigmoid_value * partial_derivative
                    bias_delta = d_sigmoid_value * partial_derivative
                    
                    reversed_weight_deltas[layer_number][previous_layer_nueron_number][current_layer_neuron_number] = weight_delta
                    reversed_bias_deltas[layer_number][previous_layer_neuron_number][0] = bias_delta
            
        return reversed(reversed_weight_deltas), reversed(reversed_bias_deltas)

    def calculateGradientsFor(self, trainingData, learningRate, sampleSize=10): 
        shuffled_data = random.shuffle(trainingData)
        while len(trainingData) > 0:
            averaged_weight_deltas = []
            averaged_bias_deltas = []
            for layer in self.layers:
                averaged_weight_deltas.append(np.zeros(layer.weightsMatrix.size))
                averaged_bias_deltas.append(np.zeros(layer.biasMatrix.size))
            
            for inputValues in shuffled_data[:sampleSize]:
                shuffled_data.pop(0) # Removes the current set of input data from shuffled_data
                expectedValues = inputData.expectedValues
                layerOutputs = forwardPropagate(inputValues, expectedValues)
                print "Cost: %0.4f" % cost(layerOutputs[-1], expectedValues)
                weight_deltas, bias_deltas = backpropagate(layerOutputs, inputValues, expectedValues)
                averaged_weight_deltas += weight_deltas
                averaged_bias_deltas += bias_deltas
            
            averaged_weight_deltas /= sampleSize
            averaged_bias_deltas /= sampleSize
        
            weight_gradient = -1.0 * learningRate * averaged_weight_deltas
            bias_gradient = -1.0 *learningRate * averaged_bias_deltas
            
        return weight_gradient, bias_gradient
    
    def train(self, trainingData, learningRate=0.2, loadLastState=False):
        if loadLastState:
            self.loadLastState()
        
        training_batch_number = 0
        try:
            while True:
                weight_gradient, bias_gradient = calculateGradientsFor(trainingData, learningRate)
                for layer_number in xrange(0, len(self.layers)):
                    self.layers[layer_number].weightsMatrix += weight_gradient[layer_number]
                    self.layers[layer_number].biasesMatrix += bias_gradient[layer_number]
                training_batch_number += 1
                print "Completed training batch number %d..." % training_batch_number
        except KeyboardInterrupt:
            print "Saving state..."
            self.saveState()
            print "Finished! Exiting now."
            System.exit(1)
    
    def saveState(self):
        try:
            os.stat(save_directory)
        except:
            os.mkdir(save_directory)
    
        for layer_number in xrange(0, len(layers)):
            layer = layers[layer_number]
            with open("%s/layer%d_weights.txt" % (save_directory, layer_number)) as weights_file:
                np.savetxt(weights_file, layer.weightsMatrix)
            with open("%s/layer%d_biases.txt" % (save_directory, layer_number)) as biases_file:
                np.savetxt(biases_file, layer.biasesMatrix)
    
    def loadLastState(self):
        try:
            os.stat(self.save_directory)
        except:
            print "Save directory \"%s\" not found. Please train the network before trying to load a state again. Thank you."
            System.exit(1)
        
        for layer_number in xrange(0, len(layers)):
            with open("%s/layer%d_weights.txt" % (save_directory, layer_number)) as weights_file:
                weightsMatrix = np.loadtxt(weights_file)
                self.layers[layer_number].weightsMatrix = weightsMatrix
            with open("%s/layer%d_biases.txt" % (save_directory, layer_number)) as biases_file:
                biasesMatrix = np.loadtxt(biases_file)
                self.layers[layer_number].biasesMatrix = biasesMatrix
    
    """
            # Calculating the output layer partial derivatives first
        output_layer_size = reversed_layers[0].size[0] # <- Gets # of rows = # of neurons
        reversed_partial_neuron_derivatives.append(np.zeros(output_layer_size, 1))
        for neuron_number in xrange(0, output_layer_size):
            neuron_value = layerOutputs[neuron_number, 0]
            expected_value = expectedValues[neuron_number]
            reversed_partial_neuron_derivatives[0].element((neuron_number, 0), neuron_value - expected_value)
    """
