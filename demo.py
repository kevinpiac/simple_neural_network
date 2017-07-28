from src.neural_network import NeuralNetwork
from numpy import array

# Instatiate our Neural Network
neural_network = NeuralNetwork()

print 'Starting Neural Network with weights:\n %s\n' %(neural_network.synaptic_weights.T)

training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T

# Set the number of iteration
training_repetition = 10000

# Train the neural network
neural_network.train(training_set_inputs, training_set_outputs, training_repetition)

print 'New weights after training:\n %s\n' %(neural_network.synaptic_weights.T)

# Predict
print 'Prediction for [1, 1, 1]: %s' %(neural_network.predict([1, 1, 1]))
