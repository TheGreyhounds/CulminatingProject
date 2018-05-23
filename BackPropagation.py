import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    sigma = sigmoid(z)
    return (1.0 - sigma) * sigma

def calculateCost(outputLayerMatrix, expectedValueMatrix):
    return np.sum(0.5 * (expectedValueMatrix - outputLayerMatrix)**2)
    
def rateOfChangeOfError(value, expectedValue, vectorSum, outputValue):
    return (expectedValue - value) * d_sigmoid(vectorSum) * outputValue


