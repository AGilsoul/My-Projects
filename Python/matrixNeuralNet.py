import random
import numpy as np
from math import sqrt, exp
import pandas as pd
from sklearn.model_selection import train_test_split


class NeuralNet:
    # Layer base class
    class Layer:
        def __init__(self, neuron_count, activation_func=None, gradient_func=None):
            self.neurons = neuron_count
            self.inputs = None
            self.pre_output = None
            self.outputs = None
            self.weights = None
            self.prev_gradients = None
            self.deltas = None
            self.biases = None
            self.prev_biases = None
            self.activation_func = activation_func
            self.gradient_func = gradient_func

        # generates random weights with xavier initialization
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
            self.pre_output = data
            return data

        # applies activation function to weighted sum
        def feed_forward(self, data):
            self.inputs = data
            self.outputs = self.activation_func(self.calc_output(data))
            return self.outputs

        def __str__(self):
            if isinstance(self, NeuralNet.InputLayer):
                return "Input Layer"
            if isinstance(self, NeuralNet.HiddenLayer):
                return "Hidden Layer"
            if isinstance(self, NeuralNet.RegressionOutputLayer):
                return "Regression Output Layer"
            if isinstance(self, NeuralNet.SoftMaxOutputLayer):
                return "Softmax Output Layer"

        def ReLu(self, output):
            for i in range(len(output)):
                if output[i] < 0:
                    output[i] = 0
            return output

        def Relu_deriv(self, data):
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

            return neuron_gradients * self.Relu_deriv(self.pre_output)

        def linear_activation(self, data):
            return np.array(data)

        def linear_gradient(self, output, target):
            return np.array([2 * (output[0] - target)])

        def softmax(self, data):
            denominator = sum(exp(x) for x in data)
            for i in range(len(data)):
                data[i] = exp(data[i]) / denominator
            return data

        # working on this still
        def softmax_deriv(self, weight_index, output, expected):
            gradients = []
            total = np.sum(self.weights, )
            return

        def cross_entropy_loss(self, output, target):
            expected = np.array(target)
            if len(expected.shape) == 1:
                expected = np.atleast_2d(expected).T

            total = - np.sum(np.log(output) * expected, axis=1)
            return total

        def softmax_cost(self, output, target):
            return np.mean(NeuralNet.Layer.cross_entropy_loss(output, target))

        @staticmethod
        def weight_derivative(delta, output):
            return delta * output

    class InputLayer(Layer):
        def __init__(self, neuron_count):
            super().__init__(neuron_count)
            self.weights = np.ones([neuron_count, neuron_count])

        def feed_forward(self, data):
            return data

    class HiddenLayer(Layer):
        def __init__(self, neuron_count):
            super().__init__(neuron_count, self.ReLu, self.ReLu_Gradient)

    class SoftMaxOutputLayer(Layer):
        def __init__(self, neuron_count):
            super().__init__(neuron_count, self.softmax, self.softmax_deriv)

    class RegressionOutputLayer(Layer):
        def __init__(self, neuron_count):
            super().__init__(neuron_count, self.linear_activation, self.linear_gradient)

    def __init__(self, lr, layer_sizes, momentum=0.0, dropout=1.0, early_stopping=-1, name="Neural Network"):
        self.lr = lr
        self.momentum = momentum
        self.dropout = dropout
        self.early_stopping = early_stopping
        self.network = np.empty(len(layer_sizes), dtype=self.Layer)
        self.build_network(layer_sizes)
        self.name = name

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

    def normalize(self, data):
        new_data = data.T
        for column in range(len(new_data)):
            col_min = min(new_data[column])
            col_max = max(new_data[column])
            for entry in range(len(new_data[column])):
                new_data[column][entry] = (new_data[column][entry] - col_min) / (col_max - col_min)
        return new_data.T

    def forward_prop(self, data):
        for i in range(len(self.network)):
            data = self.network[i].feed_forward(data)
            self.network[i].output = data
        return data

    def back_prop(self, data, targets):
        for d_index in range(len(data)):
            cur_result = self.forward_prop(train_data[d_index])
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
                        w_res = self.network[index].weight_derivative(self.network[index].deltas[n_index], self.network[index].inputs[w_index]) * self.lr
                        self.network[index].weights[n_index][w_index] -= w_res + self.network[index].prev_gradients[n_index][w_index] * self.momentum
                        self.network[index].prev_gradients[n_index][w_index] = w_res + self.network[index].prev_gradients[n_index][w_index] * self.momentum
                    b_res = self.network[index].deltas[n_index] * self.lr
                    self.network[index].biases[n_index] -= b_res + self.network[index].prev_biases[n_index] * self.momentum
                    self.network[index].prev_biases[n_index] = b_res + self.network[index].prev_biases[n_index] * self.momentum

    def train(self, data, targets, iterations):
        for i in range(iterations):
            self.back_prop(data, targets)

    def predict(self, data_instance):
        if len(self.network[-1].weights) == 1:
            return self.forward_prop(data_instance)
        else:
            prob_results = self.forward_prop(data_instance)
            return prob_results.index(max(prob_results))


    def test(self, data, targets):
        if len(self.network[-1].weights) == 1:
            model_results = [self.predict(x) for x in data]
            residual_vals = [targets[x] - model_results[x] for x in range(len(model_results))]
            residual_squared = sum([x**2 for x in residual_vals])
            average_target = sum(targets) / len(targets)
            sum_of_squares = sum([(x - average_target)**2 for x in targets])
            means_squared_error = sum([x**2 for x in residual_vals]) / len(data)
            return means_squared_error[0], 1 - (residual_squared / sum_of_squares)[0]

    def __str__(self):
        output = "\n\n" + self.name + "\n"
        output += "_____________________________________________________\n"
        output += "Learning Rate: " + str(self.lr) + " | Momentum: " + str(self.momentum) + "\n"
        output += "_____________________________________________________\n"
        output += "Input Layer Size: " + str(self.network[0].neurons) + " neurons\n"
        output += "---------------------------\n"
        for i in range(1, len(self.network) - 1):
            output += "Hidden Layer " + str(i) + " Size: " + str(self.network[i].neurons) + " neurons\n"
            output += "---------------------------\n"
        output += "Output Layer Size: " + str(self.network[-1].neurons) + " neurons\n"
        output += "_____________________________________________________\n"
        output += "Total Connections: " + str(sum(self.network[x].neurons * self.network[x - 1].neurons for x in range(1, len(self.network))))
        return output


# creates a neural network with a learning rate of 0.01, momentum of 0.9
# input layer for 8 inputs, hidden layer with 15 neurons and output with 1 neuron
net = NeuralNet(0.001, [8, 15, 1], momentum=0.0, name="COOLING LOAD")
house_data = pd.read_csv("energyefficiency.csv")

data_targets = np.array(house_data['cooling load'])
data_in = net.normalize(np.array(house_data.drop('cooling load', axis=1)))

train_data, test_data, train_targets, test_targets = train_test_split(data_in, data_targets, test_size=0.33, random_state=42)


print(net)
net.train(train_data, train_targets, 100)
MSE, r2 = net.test(test_data, test_targets)
print("MSE: " + str(MSE))
print("R^2: " + str(r2))

print(net.predict(data_in[23]))
print(data_targets[23])
