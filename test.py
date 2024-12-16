import random
import numpy as np
import gzip

class Neural_network:
    def __init__(self, input_neurons, hidden_layers, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.hidden_layers = hidden_layers

        # Initialize weights with He initialization
        self.weights = [np.random.randn(x, y) * np.sqrt(2.0/x) for x, y in zip([input_neurons] + hidden_layers, hidden_layers + [output_neurons])]
        self.biases = [np.zeros((1, y)) for y in hidden_layers + [output_neurons]]

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def feed_forward(self, x):
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(activation, w) + b
            zs.append(z)
            activation = self.relu(z)
            activations.append(activation)
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        zs.append(z)
        activation = self.softmax(z)
        activations.append(activation)
        return activations, zs

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def cross_entropy_loss(self, output_activations, y):
        epsilon = 1e-7  # Small value to avoid log(0)
        return -np.sum(y * np.log(output_activations + epsilon))

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Forward pass
        activations, zs = self.feed_forward(x)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(activations[-2].T, delta)

        for l in range(2, len(self.weights) + 1):
            z = zs[-l]
            sp = self.relu_prime(z)
            delta = np.dot(delta, self.weights[-l+1].T) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(activations[-l-1].T, delta)

        return nabla_w, nabla_b

    def train(self, training_data, eta, epochs, mini_batch_size):
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            
            for mini_batch in mini_batches:
                x_mini, y_mini = zip(*mini_batch)
                x_mini = np.array(x_mini)
                y_mini = np.array(y_mini)

                nabla_w = [np.zeros(w.shape) for w in self.weights]
                nabla_b = [np.zeros(b.shape) for b in self.biases]

                for x, y in zip(x_mini, y_mini):
                    delta_nabla_w, delta_nabla_b = self.backprop(x.reshape(1, -1), y.reshape(1, -1))
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

                self.weights = [w - (eta / mini_batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - (eta / mini_batch_size) * nb for b, nb in zip(self.biases, nabla_b)]

            # Calculate accuracy and loss after each epoch
            accuracy, loss = self.evaluate(training_data)
            print(f'Epoch {epoch + 1}: Accuracy: {accuracy:.4f}, Loss: {loss:.4f}')

    def evaluate(self, data):
        total_loss = 0
        correct_predictions = 0
        for x, y in data:
            activations, _ = self.feed_forward(x.reshape(1, -1))
            total_loss += self.cross_entropy_loss(activations[-1], y.reshape(1, -1))
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

# Neural network parameters
input_neurons = 784  # MNIST input size (28x28 pixels)
hidden_layers = [128, 64]  # Simplified architecture
output_neurons = 10   # Number of classes (digits 0-9)
eta = 0.01            # Learning rate
epochs = 10           # Number of training epochs
mini_batch_size = 32  # Mini-batch size for SGD

# Initialize and train the model
model = Neural_network(input_neurons, hidden_layers, output_neurons)
model.train(training_data, eta=eta, epochs=epochs, mini_batch_size=mini_batch_size)

# Evaluate on test data
test_accuracy, test_loss = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")