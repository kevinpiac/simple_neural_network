from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same number
        # every time the program runs.
        random.seed(1)

        # We model a single neruon, with 3 input and 1 output.
        # We assign random weights to a 3 x 1 matrix.
        # Values are in the range -1 to 1 with 0 and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # We use sigmoid function in order to range the output between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Gradient of the sigmoid curve
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, inputs):
        # prediction is the dot product of our inputs and our weights
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
        for i in xrange(number_of_iterations):
            output = self.predict(training_set_inputs)

            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of sigmoid curve
            # Less confident weights are ajudted more and input equal to 0 don't causes any change
            # It's called gradient descent
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustment
