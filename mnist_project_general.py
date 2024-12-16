import random           # This library is for randomly assigning the wights and biases 
import numpy as np      # This is the NumPy library for arrays and other functions
import gzip             # This for for unzipping the MNIST Dataset for training and testing
import warnings


class Neural_network:       # This is the class of the neural network
    def __init__(self, input_neurons, hidden_layers, output_neurons):       # This is the initializer method for the neural network
        self.input_neurons = input_neurons      # This is for the number of input neurons
        self.output_neurons = output_neurons    # This is for the number of output layers
        self.hidden_layers = hidden_layers      # This is an array containing the no. of neurons in the hidden layers

        self.weights = []   # This array is for storing the weights of the neural network
        self.biases = []    # This array is for storing the biases of the neural network

        ''' (This can also be used, this is for He initialization)
        # Initializing the weights and biases
        self.weights.append(np.random.randn(input_neurons, hidden_layers[0]) * np.sqrt(2 / input_neurons))
        self.biases.append(np.zeros(hidden_layers[0]))

        for i in range(1, len(hidden_layers)):  # Initializing the weights and biases of the hidden layers
            self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]) * np.sqrt(2 / hidden_layers[i-1]))
            self.biases.append(np.zeros(hidden_layers[i]))
        # Initializing the weights and biases of the output layer
        self.weights.append(np.random.randn(hidden_layers[-1], output_neurons) * np.sqrt(2 / hidden_layers[-1]))
        self.biases.append(np.zeros(output_neurons))'''


        # Initializing the weights and biases  (This is for the input layer and 1st hidden layer)
        self.weights.append(np.random.randn(input_neurons, hidden_layers[0]))
        self.biases.append(np.zeros(hidden_layers[0]))

        for i in range(1, len(self.hidden_layers)):     # Initializing the weights and biases of the hidden layers
            self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]))
            self.biases.append(np.zeros(hidden_layers[i]))
        # Initializing the weights and biases of the output layer
        self.weights.append(np.random.randn(hidden_layers[-1], output_neurons))
        self.biases.append(np.zeros(output_neurons))


    def sigmoid(self, x):   # This is the function for the sigmoid function (Non-Linear Activation Function)
        warnings.filterwarnings('ignore', 'overflow')            # These lines are for avoiding the warnings
        warnings.filterwarnings('ignore', '(overflow|invalid)')
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_prime(self, x): # This function is for the derivative of the sigmoid function
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def feed_forward(self, x):  # This is my feed-forward function for getting the activations of the neurons in order to update the weights and biases in the backpropagation part
        activations = [x]   # This is the activations array for storing the activation values of the neurons
        zs = []             # This is for storing the values before applying the activation function to the output of the neurons

        for i in range(len(self.hidden_layers)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(z)
        output_activation = self.sigmoid(z)
        activations.append(output_activation)

        return activations, zs      # Returning the activations and the values before the activations

    def segregate_batches(self, training_data, mini_batch_size):    # This function is for making the training data into mini batches
        random.shuffle(training_data)
        mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
        return mini_batches

    def train(self, training_data, eta, epochs, mini_batch_size):   # This is the training function
        for epoch in range(epochs):
            random.shuffle(training_data)       # Randomly shuffling the data before segregating
            mini_batches = self.segregate_batches(training_data, mini_batch_size)
            for mini_batch in mini_batches:
                x_mini, y_mini = zip(*mini_batch)
                x_mini = np.array(x_mini)           # This is the training input
                y_mini = np.array(y_mini)           # This is the label (desired output)

                nabla_w = [np.zeros(w.shape) for w in self.weights]     # This is the array storing the delta wrt weight (for whole mini batch)
                nabla_b = [np.zeros(b.shape) for b in self.biases]      # This is the array storing the delta wrt weight

                for x, y in zip(x_mini, y_mini):
                    delta_nabla_w, delta_nabla_b = self.backprop(x, y)      # Here I have the deltas returned from the back propagation function (these are for single training example)
                    # Below I have summed the deltas for the whole mini batch
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

                self.weights = [w - (eta / mini_batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - (eta / mini_batch_size) * nb for b, nb in zip(self.biases, nabla_b)]

            # Calculate accuracy and loss after each epoch
            accuracy, loss = self.evaluate(training_data)
            print(f'Epoch {epoch + 1}: Accuracy: {accuracy:.4f}, Loss: {loss:.4f}')

    def backprop(self, x, y):       # This is the function responsible for the application of the "Stochastic Gradient Descent Algorithm"
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activations, zs = self.feed_forward(x)

        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])        # This delta is the partial derivative of C wrt zL
        nabla_w[-1] = np.dot(activations[-2].reshape(-1, 1), delta.reshape(1, -1))
        nabla_b[-1] = delta

        for l in range(2, len(self.hidden_layers) + 2):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(delta, self.weights[-l + 1].T) * sp
            nabla_w[-l] = np.dot(activations[-l - 1].reshape(-1, 1), delta.reshape(1, -1))
            nabla_b[-l] = delta

        return nabla_w, nabla_b

    def cost_derivative(self, output_activations, y):   # This function finds the difference between desired and calculated output
        return output_activations - y

    def evaluate(self, data):       # This is for evaluating the model
        total_loss = 0
        correct_predictions = 0
        for x, y in data:
            activations, _ = self.feed_forward(x)
            total_loss += 0.5 * np.sum((activations[-1] - y) ** 2)
            correct_predictions += int(np.argmax(activations[-1]) == np.argmax(y))
        accuracy = correct_predictions / len(data)
        return accuracy, total_loss / len(data)

def load_mnist():       # This is for loading the MNIST Dataset into the model
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 28 * 28) / 255.0
        return data

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

   
    train_images = load_images('train-images-idx3-ubyte.gz')
    train_labels = load_labels('train-labels-idx1-ubyte.gz')
    test_images = load_images('t10k-images-idx3-ubyte.gz')
    test_labels = load_labels('t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels

# Load the data
X_train, y_train, X_test, y_test = load_mnist()

# One-hot encode the labels
y_train_one_hot = np.eye(10)[y_train]
y_test_one_hot = np.eye(10)[y_test]

# Prepare the training data as list of tuples
training_data = list(zip(X_train, y_train_one_hot))
test_data = list(zip(X_test, y_test_one_hot))

# Neural network parameters (These can be changed based on our use)
input_neurons = 784  # MNIST input size (28x28 pixels)
hidden_layers = [10, 15, 10, 15, 20]  # Adjust based on experimentation
output_neurons = 10   # Number of classes (digits 0-9)
eta = 0.8           # Starting learning rate
epochs = 15           # Adjust based on convergence
mini_batch_size = 32  # Typical mini-batch size for SGD

# Initialize and train the model
model = Neural_network(input_neurons, hidden_layers, output_neurons)
model.train(training_data, eta=eta, epochs=epochs, mini_batch_size=mini_batch_size)

# For the np.exp() overflow warning we can either use the float 128, but it requires much more memory and this slows our model, or we can just ignore the warning for the overflow