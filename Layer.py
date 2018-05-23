import numpy.matlib as np
from numpy import exp

class Layer:
    def __init__(self, size, previousLayerSize):
        self.size = size
        self.previousLayerSize = previousLayerSize
        self.weightsMatrix = np.zeroes(size, previousLayerSize)
        self.biasesMatrix = np.zeroes(size, 1)
    
    def sigmoid(matrix):
        return 1.0 / (1.0 + exp(-1.0*matrix))
    
    def generateNextLayerOutput(lastLayerOutputMatrix):
        return sigmoid(np.cross(self.weightsMatrix, lastLayerOutputMatrix) + self.biasesMatrix)
    
    # weightIndex must be a tuple with weightIndex[0] representing
    # the row index and weightIndex[1] representing the column index.
    def changeWeight(self, weightIndex, changeWeightBy):
        try:
            originalWeight = self.weightMatrix[weightIndex[0], weightIndex[1]]
            self.weightMatrix[weightIndex[0], weightIndex[1]] = originalWeight + changeWeightBy
        except IndexError:
            print "Exception. weightIndex \"" weightIndex "\" is out of the range of the matrix."
            print "The first value must be between 0 and " + self.previousLayerSize + " while"
            print "The second value must be between 0 and " + self.size + "."
            sys.exit(1)
    
    def changeBias(self, biasIndex, changeBiasBy):
        try:
        if 0 <= biasIndex and biasIndex < size:
            originalBias = self.biasMatrix[biasIndex, 0]
            self.biasMatrix[biasIndex, 0] = originalBias + changeBiasBy
        except IndexError:
            print "Exception. biasIndex \"" biasIndex "\" is out of the range of the matrix."
            print "The biasIndex value must be between 0 and " + self.size + "."
            sys.exit(1)
    
