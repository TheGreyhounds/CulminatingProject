class Node:
    def __init__(self, weightslayerBehind, weightslayerAhead):
        self.biasValue = 0.0
        self.weightslayerBehind = layerBehind
        self.weightslayerAhead = layerAhead
    #setters:
    def setBias(biasValue):
        if 0 <= biasValue and biasValue <= 1:
            self.biasValue = biasValue
            return True
        return False
    #getters:
    def getBias():
        return self.biasValue
    def getInput():
        return self.inputValue
