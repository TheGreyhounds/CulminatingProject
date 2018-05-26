import mnist
from Layer import Layer
from Network import Network

hidden_layer_1 = Layer(28, 784)
hidden_layer_2 = Layer(16, 28)
output_layer = Layer(10, 16)

layers = [hidden_layer_1, hidden_layer_2, output_layer]

network = Network(layers)

image_generator = mnist.read()
image = image_generator.next()

mnist.show(image[1])
