class Node:
    def __init__(self):
        self.biasValue = 0.0
        self.inputValue = 0.0
        self.nextNodeList = [None]
    #setters:
    def setBias(biasValue):
        if (0 <= biasValue <= 1):
            self.biasValue = biasValue
            return True
        return False
    def setInput(inputValue):
        self.inputValue = inputValue
    def addNewNode(newNode):
        self.nextNodeList.append(newNode)
    #getters:
    def getBias():
        return self.biasValue
    def getInput():
        return self.inputValue
    def getNode(index):
    	if (index >= len(self.nextNodeList)):
    	    return None
        return self.nextNodeList[index]