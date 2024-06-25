import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error loss function
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Define the neural network class
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)  # weights for the hidden layer
        self.weights2 = np.random.rand(4, 1)  # weights for the output layer
        self.y = y
        self.output = np.zeros(self.y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        
    def backprop(self):
        # Calculate the error
        error = self.y - self.output
        d_weights2 = np.dot(self.layer1.T, (2 * error * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2 * error * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        
        # Update the weights with the derivative of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, epochs=10000):
        for _ in range(epochs):
            self.feedforward()
            self.backprop()

# Define the input data (XOR dataset)
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize and train the neural network
nn = NeuralNetwork(x, y)
nn.train(epochs=10000)

# Test the neural network
nn.feedforward()
print("Predicted Output:")
print(nn.output)
print("Actual Output:")
print(y)

# Calculate the loss
loss = mse_loss(y, nn.output)
print("Loss:", loss)
