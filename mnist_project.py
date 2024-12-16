import random
import numpy as np
import gzip

class Neural_Network:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        # Initialize weights and biases with He initialization
        self.weights_hidden = np.random.randn(input_neurons, hidden_neurons)
        self.biases_hidden = np.random.randn(hidden_neurons)

        self.weights_output = np.random.randn(hidden_neurons, output_neurons)
        self.biases_output = np.random.randn(output_neurons)

    def feed_forward(self, x):
        # computing the output of hidden layer:
        hidden_activation = self.sigmoid(np.dot(x, self.weights_hidden) + self.biases_hidden)

        # computing the output of output layer:
        output_activation = self.sigmoid(np.dot(hidden_activation, self.weights_output) + self.biases_output)

        return hidden_activation, output_activation

    def segregate_batches(self, training_data, mini_batch_size):
        random.shuffle(training_data)
        mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
        return mini_batches

    def cost_calculation(self, output_activation, y):
        return output_activation - y

    def train(self, training_data, eta, epochs, mini_batch_size, test_data=None):
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = self.segregate_batches(training_data, mini_batch_size)
            for mini_batch in mini_batches:
                x_mini, y_mini = zip(*mini_batch)
                x_mini = np.array(x_mini)
                y_mini = np.array(y_mini)

                hidden_activations, output_activations = self.feed_forward(x_mini)

                # Calculate output delta using sigmoid prime and average over mini-batch
                output_delta = np.mean((output_activations - y_mini) * output_activations * (1 - output_activations), axis=0)

                # Calculate gradients
                dw_output = np.outer(hidden_activations.T, output_delta) / mini_batch_size
                db_output = np.sum(output_delta, axis=0) / mini_batch_size

                hidden_delta = np.dot(output_delta, self.weights_output.T) * hidden_activations * (1 - hidden_activations)
                dw_hidden = np.dot(x_mini.T, hidden_delta) / mini_batch_size
                db_hidden = np.sum(hidden_delta, axis=0) / mini_batch_size

                # Update weights and biases
                self.weights_hidden -= eta * dw_hidden
                self.biases_hidden -= eta * db_hidden
                self.weights_output -= eta * dw_output  # Check this line
                self.biases_output -= eta * db_output

            # Evaluate accuracy after each epoch if test_data provided
            if test_data:
                accuracy = self.evaluate(test_data)
                print(f'Epoch {epoch}: Accuracy: {accuracy:.2%}')


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)[1]), np.argmax(y)) for x, y in test_data]
        accuracy = sum(int(predicted == actual) for predicted, actual in test_results) / len(test_data)
        return accuracy

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
hidden_neurons = 100  # Adjust based on experimentation
output_neurons = 10   # Number of classes (digits 0-9)
eta = 0.5           # Starting learning rate
epochs = 15           # Adjust based on convergence
mini_batch_size = 32  # Typical mini-batch size for SGD

# Initialize and train the model
model = Neural_Network(input_neurons, hidden_neurons, output_neurons)
model.train(training_data, eta=eta, epochs=epochs, mini_batch_size=mini_batch_size, test_data=test_data)

# Evaluate the final model accuracy
accuracy = model.evaluate(test_data)
print(f'Final Accuracy: {accuracy:.4f}')
