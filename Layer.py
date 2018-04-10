import numpy.matlib as np

class Layer:
    def __init__(self, size):
        self.size = size
        self.nodeMatrix = np.zeroes(size, 1)
    
    def setWeightOf(nodeNumber, newWeight, layerAhead=True, connectingNodeNumber):
        if 0 <= nodeNumber and nodeNumber <= size:
             node = self.nodeMatrix.item((nodeNumber, 1))
             if layerAhead:
                 node.layerAhead[connectingNodeNumber]
                 self.nodeMatrix.itemset((nodeNumber, 1), node.layerAhead[connectingNodeNumber]
    
    def getNode(index):
    	if (index >= len(self.nextNodeList)):
    	    return None
        return self.nextNodeList[index]
    
    def addNewNode(newNode):
        self.nodeMatrix.append(newNode)    
    
