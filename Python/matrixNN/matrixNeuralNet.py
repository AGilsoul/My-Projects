import keras.utils.np_utils
import numpy as np
from math import sqrt, exp
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


class NeuralNet:
    # Layer super class
    class Layer:
        # layer constructor
        def __init__(self, neuron_count, activation_func=None, gradient_func=None):
            self.neurons = neuron_count
            self.inputs = None
            self.pre_outputs = None
            self.activated_outputs = None
            self.weights = None
            self.prev_gradients = None
            self.deltas = None
            self.biases = None
            self.prev_biases = None
            self.activation_func = activation_func
            self.gradient_func = gradient_func

        # generates random weights with normalized xavier weight initialization
        def generate_weights(self, num_out, num_weights):
            lower, upper = -(1.0 * sqrt(6.0) / sqrt(num_weights + num_out)), (1.0 * sqrt(6.0) / sqrt(num_weights + num_out))
            result_weights = np.random.rand(num_out, num_weights)
            result_weights = lower + result_weights * (upper - lower)
            self.biases = np.zeros(num_out)
            self.prev_biases = np.zeros(num_out)
            self.weights = result_weights
            self.prev_gradients = np.zeros([num_out, num_weights])

        # weighted sum + bias for a layer
        def calc_output(self, data):
            data = np.dot(self.weights, data)
            for i in range(len(data)):
                data[i] += self.biases[i]
            self.pre_outputs = data
            return data

        # applies activation function to weighted sum
        def feed_forward(self, data):
            self.inputs = data
            self.activated_outputs = self.activation_func(self.calc_output(data))
            return self.activated_outputs

        # string method for layer class
        def __str__(self):
            if isinstance(self, NeuralNet.InputLayer):
                return "Input Layer"
            if isinstance(self, NeuralNet.HiddenLayer):
                return "Hidden Layer"
            if isinstance(self, NeuralNet.RegressionOutputLayer):
                return "Regression Output Layer"
            if isinstance(self, NeuralNet.SoftMaxOutputLayer):
                return "Softmax Output Layer"

        @staticmethod
        def ReLu(output):
            for i in range(len(output)):
                if output[i] < 0:
                    output[i] = 0
            return output

        @staticmethod
        def Relu_deriv(data):
            result = np.zeros([len(data)], dtype=float)
            for i in range(len(data)):
                if data[i] > 0:
                    result[i] = 1
                else:
                    result[i] = 0
            return result

        def ReLu_Gradient(self, next_deltas, next_layer):
            neuron_gradients = np.zeros([len(self.weights)], dtype=float)
            for weight_index in range(len(self.weights)):
                total = 0
                for n in range(len(next_layer.weights)):
                    total += next_layer.weights[n][weight_index] * next_deltas[n]
                neuron_gradients[weight_index] = total

            return neuron_gradients * self.Relu_deriv(self.pre_outputs)

        @staticmethod
        def linear_activation(data):
            return np.array(data)

        @staticmethod
        def linear_gradient(output, target):
            return np.array([2 * (output[0] - target)])

        @staticmethod
        def softmax(data):
            denominator = sum(exp(x) for x in data)
            for i in range(len(data)):
                data[i] = (exp(data[i]) / denominator)
            return data

        # working on this still
        def softmax_deriv(self):
            denominator = sum(exp(x) for x in self.pre_outputs)
            gradients = []
            for i in range(len(self.pre_outputs)):
                gradients.append((exp(self.pre_outputs[i]) * denominator - exp(self.pre_outputs[i])**2)/(denominator**2))
            return np.array(gradients)

        def softmax_gradient(self, outputs, targets):
            # resulting_gradients = []
            # soft_derivs = self.softmax_deriv()
            # for i in range(len(targets)):
                # resulting_gradients.append(-1 * (targets[i] * (1 / self.activated_outputs[i]) * soft_derivs[i]))
            # return np.array(resulting_gradients)
            return outputs - targets

        @staticmethod
        def cross_entropy_loss(output, target):
            expected = np.array(target)
            if len(expected.shape) == 1:
                expected = np.atleast_2d(expected).T

            total = - np.sum(np.log(output) * expected, axis=1)
            return total

        @staticmethod
        def softmax_cost(output, target):
            return np.mean(NeuralNet.Layer.cross_entropy_loss(output, target))

        @staticmethod
        def mean_squared_error(outputs, targets):
            return np.mean(np.array([(outputs[x] - targets[x])**2 for x in range(len(outputs))]))

        @staticmethod
        def weight_derivative(delta, output):
            return delta * output

    # Input layer subclass
    class InputLayer(Layer):
        def __init__(self, neuron_count):
            super().__init__(neuron_count)
            self.weights = np.ones([neuron_count, neuron_count])

        def feed_forward(self, data):
            return data

    # Hidden layer subclass
    class HiddenLayer(Layer):
        def __init__(self, neuron_count):
            super().__init__(neuron_count, self.ReLu, self.ReLu_Gradient)

    # Softmax layer subclass
    class SoftMaxOutputLayer(Layer):
        def __init__(self, neuron_count):
            super().__init__(neuron_count, self.softmax, self.softmax_gradient)

    # Regression layer subclass
    class RegressionOutputLayer(Layer):
        def __init__(self, neuron_count):
            super().__init__(neuron_count, self.linear_activation, self.linear_gradient)

    # Network constructor
    # learning rate and layer sizes are required
    # momentum, dropout, early stopping, name, and graphing are all default
    def __init__(self, lr, layer_sizes, momentum=0.0, dropout=1.0, early_stopping=-1, weight_decay=0.0, name="Neural Network", graph=False):
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dropout = dropout
        self.early_stopping = early_stopping
        self.network = np.empty(len(layer_sizes), dtype=self.Layer)
        self.build_network(layer_sizes)
        self.name = name
        self.graph = graph
        self.conversions = False
        self.conversion_rates = None
        self.prev_trained = False

    # builds the network using the given n layer sizes, creates one input layer, n-2 hidden layers, and one output layer
    # whose type is determined by the number the size of the output layer
    # more than one neuron will use softmax, one neuron will use regression
    def build_network(self, layer_sizes):
        self.network[0] = self.InputLayer(layer_sizes[0])
        for i in range(1, len(layer_sizes) - 1):
            self.network[i] = self.HiddenLayer(layer_sizes[i])
            self.network[i].generate_weights(layer_sizes[i], layer_sizes[i-1])
        if layer_sizes[-1] > 1:
            self.network[-1] = self.SoftMaxOutputLayer(layer_sizes[-1])
        else:
            self.network[-1] = self.RegressionOutputLayer(1)
        self.network[-1].generate_weights(layer_sizes[-1], layer_sizes[-2])

    # min-max normalization for a dataset
    # val-ranges can be set if every value is in the same known range (pixel values all between 0-255, etc.)
    def normalize(self, data, val_ranges=None):
        data = np.array(data)
        if not self.conversions:
            if val_ranges:
                for row in range(len(data)):
                    print(row)
                    for column in range(len(data[row])):
                        data[row][column] = (data[row][column] - val_ranges[0]) / (val_ranges[1] - val_ranges[0])
                self.conversion_rates = np.array([[val_ranges[0], val_ranges[1]] for _ in data[0]])
                self.conversions = True
                return data
            else:
                new_data = data.T
                all_rates = []
                for column in range(len(new_data)):
                    col_min = min(new_data[column])
                    col_max = max(new_data[column])
                    for entry in range(len(new_data[column])):
                        new_data[column][entry] = (new_data[column][entry] - col_min) / (col_max - col_min)
                    all_rates.append([col_min, col_max])
                self.conversion_rates = np.array(all_rates)
                self.conversions = True
                return new_data.T
        else:
            new_data = data.T
            for column in range(len(new_data)):
                col_min = self.conversion_rates[column][0]
                col_max = self.conversion_rates[column][1]
                for entry in range(len(new_data[column])):
                    new_data[column][entry] = (new_data[column][entry] - col_min) / (col_max - col_min)
            return new_data.T

    # forward propagation method, uses each layers feed forward method output as the input for the next layer
    def __forward_prop(self, data):
        for i in range(len(self.network)):
            data = self.network[i].feed_forward(data)
            self.network[i].output = data
        return data

    # back propagation method, updates the weights and biases of each neuron in each layer
    def __back_prop(self, data, targets):
        for d_index in range(len(data)):
            cur_result = self.__forward_prop(data[d_index])
            cur_label = targets[d_index]
            # output layer delta computation
            next_deltas = self.network[-1].gradient_func(cur_result, cur_label)
            self.network[-1].deltas = next_deltas

            # hidden layer delta comp
            for index in reversed(range(1, len(self.network) - 1)):
                next_deltas = self.network[index].gradient_func(next_deltas, self.network[index+1])
                self.network[index].deltas = next_deltas

            # update layer weights and biases
            for index in reversed(range(1, len(self.network))):
                for n_index in range(len(self.network[index].weights)):
                    for w_index in range(len(self.network[index].weights[n_index])):
                        weight_val = self.network[index].weights[n_index][w_index]
                        w_deriv = self.network[index].weight_derivative(self.network[index].deltas[n_index], self.network[index].inputs[w_index]) + (2 * self.weight_decay * weight_val)
                        gradient = w_deriv * self.lr + (self.network[index].prev_gradients[n_index][w_index] * self.momentum)
                        self.network[index].weights[n_index][w_index] -= gradient
                        self.network[index].prev_gradients[n_index][w_index] = gradient
                    b_res = self.network[index].deltas[n_index] * self.lr
                    self.network[index].biases[n_index] -= b_res + self.network[index].prev_biases[n_index] * self.momentum
                    self.network[index].prev_biases[n_index] = b_res + self.network[index].prev_biases[n_index] * self.momentum

    def __back_prop_minibatch(self, data, targets):
        return

    # training method, repeats back propagation for the max iterations
    # if min iterations and validation data is provided, will include early stopping in the process
    # if graphing is enabled, will display a graph of validation accuracy over time
    def train(self, training_data, training_targets, max_iterations, min_iterations=0, validation_data=None, validation_targets=None):
        if self.prev_trained:
            self.reset_for_training()
        if self.early_stopping == -1:
            for i in range(max_iterations):
                print(i)
                self.__back_prop(training_data, training_targets)
        elif len(self.network[-1].weights) == 1 and self.early_stopping != -1 and (validation_data is not None or validation_targets is not None):
            max_r_squared = 0.0
            decrease_streak = 0
            val_results = []
            train_results = []
            for i in range(max_iterations):
                self.__back_prop(training_data, training_targets)
                train_res = self.test(training_data, training_targets)[1]
                test_res = self.test(validation_data, validation_targets)[1]
                train_results.append(train_res)
                val_results.append(test_res)
                if test_res > max_r_squared:
                    max_r_squared = test_res
                    decrease_streak = 0
                elif i >= min_iterations:
                    decrease_streak += 1
                if decrease_streak >= self.early_stopping:
                    break
            if self.graph:
                plt.plot(range(1, len(val_results)+1), val_results, label='Validation')
                plt.plot(range(1, len(train_results)+1), train_results, label='Training')
                plt.ylabel("R^2")
                plt.xlabel("Iterations")
                plt.title("Training/Validation Accuracy")
                plt.legend()
                plt.show()
        elif len(self.network[-1].weights) != 1 and self.early_stopping != -1 and (validation_data is not None or validation_targets is not None):
            max_accuracy = 0.0
            decrease_streak = 0
            train_results = []
            val_results = []
            for i in range(max_iterations):
                self.__back_prop(training_data, training_targets)
                train_res = self.test(training_data, training_targets)
                test_res = self.test(validation_data, validation_targets)
                train_results.append(train_res)
                val_results.append(test_res)
                if test_res > max_accuracy:
                    max_accuracy = test_res
                    decrease_streak = 0
                elif i >= min_iterations:
                    decrease_streak += 1
                if decrease_streak >= self.early_stopping:
                    break
            if self.graph:
                plt.plot(range(1, len(val_results) + 1), val_results, label='Validation')
                plt.plot(range(1, len(train_results) + 1), train_results, label='Training')
                plt.ylabel("Accuracy")
                plt.xlabel("Iterations")
                plt.title("Training/Validation Accuracy")
                plt.show()
        self.prev_trained = True

    # prediction method for both categorical and numerical outputs
    # if model is for regression, returns the single output value of the output layer
    # if model is for classification, returns the index of the neuron with the highest probability (softmax value)
    def __predict(self, data_instance):
        if len(self.network[-1].weights) == 1:
            return self.__forward_prop(data_instance)
        else:
            prob_results = self.__forward_prop(data_instance)
            return np.where(prob_results == max(prob_results))

    def predict(self, data):
        data = self.normalize(np.array([data]))[0]
        return self.__predict(data)[0]

    # testing method
    # for regression, returns the MSE and R^2 values
    # for classification, returns the accuracy (decimal, not percentage) of testing samples correctly classified
    def test(self, data, targets):
        if len(self.network[-1].weights) == 1:
            model_results = [self.__predict(x) for x in data]
            residual_vals = [targets[x] - model_results[x] for x in range(len(model_results))]
            residual_squared = sum([x**2 for x in residual_vals])
            average_target = sum(targets) / len(targets)
            sum_of_squares = sum([(x - average_target)**2 for x in targets])
            mean_squared_error = sum([x**2 for x in residual_vals]) / len(data)
            return mean_squared_error[0], 1 - (residual_squared / sum_of_squares)[0]
        else:
            num_correct = 0
            model_results = [self.__predict(x)[0] for x in data]
            target_indices = [x.index(1) for x in targets]
            for i in range(len(model_results)):
                if model_results[i] == target_indices[i]:
                    num_correct += 1
            return num_correct / len(targets)

    def reset_for_training(self):
        for layer in range(len(self.network)):
            for n in range(len(self.network[layer].prev_gradients)):
                self.network[layer].prev_biases[n] = 0
                for g in range(len(self.network[layer].prev_gradients[n])):
                    self.network[layer].prev_gradients[n][g] = 0

    # string method for NeuralNet object
    # returns a string including the network name, layers, and layer sizes, along with the total connections
    def __str__(self):
        output = "\n\n" + self.name + "\n"
        output += "_____________________________________________________\n"
        output += "Learning Rate: " + str(self.lr)
        if self.momentum != 0.0:
            output += " | Momentum: " + str(self.momentum)
        if self.dropout != 1.0:
            output += " | Dropout: " + str(self.dropout)
        if self.early_stopping != -1:
            output += " | Early Stopping: " + str(self.early_stopping)
        output += "\n"
        output += "_____________________________________________________\n"
        output += str(self.network[0]) + " Size: " + str(self.network[0].neurons) + " neurons\n"
        output += "---------------------------\n"
        for i in range(1, len(self.network) - 1):
            output += str(self.network[i]) + " " + str(i) + " Size: " + str(self.network[i].neurons) + " neurons\n"
            output += "---------------------------\n"
        output += str(self.network[-1]) + " Size: " + str(self.network[-1].neurons) + " neuron(s)\n"
        output += "_____________________________________________________\n"
        output += "Total Connections: " + str(sum(self.network[x].neurons * self.network[x - 1].neurons for x in range(1, len(self.network))))
        return output


def energy_config():
    # creates a neural network with a learning rate of 0.001, early stopping at 15 iterations, and graphing enabled
    # input layer for 8 inputs, two hidden layers with 15 neurons and output layer with 1 neuron
    net = NeuralNet(0.001, [8, 15, 15, 1], early_stopping=15, name="COOLING LOAD", graph=True)
    # loads house data
    house_data = pd.read_csv("energyefficiency.csv")
    # creates target array
    data_targets = np.array(house_data['cooling load'])
    # normalizes input data
    data_in = np.array(house_data.drop('cooling load', axis=1))
    # splits the dataset into training, validation, and testing datasets
    train_data, temp_data, train_targets, temp_targets = train_test_split(data_in, data_targets, test_size=0.6)
    val_data, test_data, val_targets, test_targets = train_test_split(temp_data, temp_targets, test_size=0.5)
    # normalizes all datasets
    train_data = net.normalize(train_data)
    val_data = net.normalize(val_data)
    test_data = net.normalize(test_data)
    # prints the network to the console
    print(net)
    # trains the network between 100-1000 iterations with early stopping enabled
    net.train(train_data, train_targets, 1000, min_iterations=250, validation_data=val_data, validation_targets=val_targets)
    # tests the fit of the neural network to the testing data
    MSE, r2 = net.test(test_data, test_targets)
    # prints testing results to the console
    print("MSE: " + str(MSE))
    print("R^2: " + str(r2))


def tumor_config():
    # creates a neural network with a learning rate of 0.0001
    # input layer for 30 inputs, hidden layer with 15 neurons and output with 2 neurons
    net = NeuralNet(0.01, [30, 15, 2], early_stopping=5, name="MALIGNANCY", graph=True)
    print(net)
    bc = load_breast_cancer()
    all_data = bc['data']
    all_data = net.normalize(all_data)
    all_targets = bc['target']
    new_targets = [[1, 0] if x == 0 else [0, 1] for x in all_targets]

    train_data, temp_data, train_targets, temp_targets = train_test_split(all_data, new_targets, test_size=0.6)
    val_data, test_data, val_targets, test_targets = train_test_split(temp_data, temp_targets, test_size=0.5)
    net.train(train_data, train_targets, 500, min_iterations=20, validation_data=val_data, validation_targets=val_targets)
    accuracy = net.test(test_data, test_targets)
    print("{:.2f}%".format(accuracy * 100))


energy_config()
# tumor_config()
