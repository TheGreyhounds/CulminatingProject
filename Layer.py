import numpy.matlib as np
from numpy import exp

def sigmoid(matrix):
        return (np.zeros(matrix.size).fill(1.0) + exp(-1.0*matrix))**(-1)
        
def d_sigmoid(matrix):
    sigma = sigmoid(matrix)
    return (np.zeros(sigma.size).fill(1.0) - sigma) * sigma        


class Layer:
    def __init__(self, size, previousLayerSize, loadState=None):
        self.size = size
        self.previousLayerSize = previousLayerSize
        if loadState is not None:
            self.weightsMatrix = loadState[0]
            self.biasesMatrix = loadState[1]
        else:
            self.weightsMatrix = np.random.rand(size, previousLayerSize)
            self.biasesMatrix = np.random.rand(size, 1)
    
    def generateNextLayerOutput(lastLayerOutputMatrix):
        return sigmoid(np.dot(self.weightsMatrix, lastLayerOutputMatrix) + self.biasesMatrix)
    
    def generateSigmoidDerivativeNextLayerOutput(lastLayerOutputMatrix):
            return sigmoid(np.dot(self.weightsMatrix, lastLayerOutputMatrix) + self.biasesMatrix)
    
    def getWeight(self, indices):
        return self.weightsMatrix[indices[0]][indices[1]]
    
    def getBias(self, indices):
        return self.biasesMatrix[indices[0]][indices[1]]
    
