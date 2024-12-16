import random           # For randomly assigning the weights and biases
import numpy as np      # For arrays and other numerical functions
import gzip             # For unzipping the MNIST dataset
import csv              # For saving results to CSV files
import warnings

class Neural_network:
    def __init__(self, input_neurons, hidden_layers, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.hidden_layers = hidden_layers

        self.weights = []
        self.biases = []

        # Initializing weights and biases (input layer and first hidden layer)
        self.weights.append(np.random.randn(input_neurons, hidden_layers[0]))
        self.biases.append(np.zeros(hidden_layers[0]))

        for i in range(1, len(self.hidden_layers)):  # Hidden layers
            self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]))
            self.biases.append(np.zeros(hidden_layers[i]))

        # Output layer
        self.weights.append(np.random.randn(hidden_layers[-1], output_neurons))
        self.biases.append(np.zeros(output_neurons))

    def sigmoid(self, x):
        warnings.filterwarnings('ignore', 'overflow')
        warnings.filterwarnings('ignore', '(overflow|invalid)')
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def feed_forward(self, x):
        activations = [x]
        zs = []

        for i in range(len(self.hidden_layers)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(z)
        output_activation = self.sigmoid(z)
        activations.append(output_activation)

        return activations, zs

    def segregate_batches(self, training_data, mini_batch_size):
        random.shuffle(training_data)
        mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
        return mini_batches

    def train(self, training_data, eta, epochs, mini_batch_size, config_name):
        results = []  # Store accuracy and loss for each epoch
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = self.segregate_batches(training_data, mini_batch_size)
            for mini_batch in mini_batches:
                x_mini, y_mini = zip(*mini_batch)
                x_mini = np.array(x_mini)
                y_mini = np.array(y_mini)

                nabla_w = [np.zeros(w.shape) for w in self.weights]
                nabla_b = [np.zeros(b.shape) for b in self.biases]

                for x, y in zip(x_mini, y_mini):
                    delta_nabla_w, delta_nabla_b = self.backprop(x, y)
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

                self.weights = [w - (eta / mini_batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - (eta / mini_batch_size) * nb for b, nb in zip(self.biases, nabla_b)]

            accuracy, loss = self.evaluate(training_data)
            print(f'Epoch {epoch + 1}: Accuracy: {accuracy:.4f}, Loss: {loss:.4f}')
            results.append([epoch + 1, accuracy, loss])

        # Save results to a CSV file
        with open(f'{config_name}_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Accuracy', 'Loss'])
            writer.writerows(results)

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activations, zs = self.feed_forward(x)

        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(activations[-2].reshape(-1, 1), delta.reshape(1, -1))
        nabla_b[-1] = delta

        for l in range(2, len(self.hidden_layers) + 2):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(delta, self.weights[-l + 1].T) * sp
            nabla_w[-l] = np.dot(activations[-l - 1].reshape(-1, 1), delta.reshape(1, -1))
            nabla_b[-l] = delta

        return nabla_w, nabla_b

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def evaluate(self, data):
        total_loss = 0
        correct_predictions = 0
        for x, y in data:
            activations, _ = self.feed_forward(x)
            total_loss += 0.5 * np.sum((activations[-1] - y) ** 2)
            correct_predictions += int(np.argmax(activations[-1]) == np.argmax(y))
        accuracy = correct_predictions / len(data)
        return accuracy, total_loss / len(data)

def load_mnist():
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

# Neural network parameters (These can be changed based on experimentation)
input_neurons = 784  # MNIST input size (28x28 pixels)
output_neurons = 10   # Number of classes (digits 0-9)
eta = 0.8           # Adjust based on convergence

# Example configuration runs
configurations = [
    ([10, 10], 10, 32, 'config1_ep'),
    ([10, 10], 12, 32, 'config2_ep'),
    ([10, 10], 15, 32, 'config3_ep'),
    ([10, 10], 20, 32, 'config4_ep')
]

for hidden_layers, epochs, mini_batch_size, config_name in configurations:
    model = Neural_network(input_neurons, hidden_layers, output_neurons)
    model.train(training_data, eta=eta, epochs=epochs, mini_batch_size=mini_batch_size, config_name=config_name)
