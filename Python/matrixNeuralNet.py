import numpy as np
from math import sqrt, exp


class NeuralNet:
    class Layer:
        def __init__(self, neuron_count, activation=None, activation_deriv=None):
            self.neurons = neuron_count
            self.input = None
            self.prev_input = None
            self.weights = None
            self.prev_gradients = None
            self.bias = 1
            self.prev_bias = 0
            self.activation = activation
            self.activation_deriv = activation_deriv

        # generates random weights with xavier initialization
        def generate_weights(self, num_weights, prev_weights):
            lower, upper = -(1.0 / sqrt(prev_weights)), (1.0 / sqrt(prev_weights))
            result_weights = np.random.rand(num_weights, prev_weights)
            result_weights = lower + result_weights * (upper - lower)
            self.weights = result_weights
            self.prev_gradients = np.zeros(num_weights)

        def calc_output(self, data):
            data = np.dot(self.weights, data)
            for i in data:
                i += self.bias
            return data

        def feed_forward(self, data):
            return self.activation(self.calc_output(data))

        def __str__(self):
            output = "Weights:\n"
            for i in range(len(self.weights)):
                output += "\tNeuron " + str(i) + ": " + str(self.weights[i]) + " | Bias: " + str(self.bias)
                output += "\n"
            return output

        @staticmethod
        def ReLu(output):
            for i in output:
                if i < 0:
                    i = 0
            return output

        @staticmethod
        def Relu_deriv(data):
            for i in data:
                if i > 0:
                    i = 1
                else:
                    i = 0
            return data

        @staticmethod
        def linear_activation(data):
            return data

        @staticmethod
        def linear_deriv(data):
            return np.ones(data.shape)

        @staticmethod
        def softmax(data):
            denominator = sum(exp(x) for x in data)
            for i in data:
                i = exp(i) / denominator
            return data

        # working on this still
        def softmax_deriv(self, weight_index, output, expected):
            gradients = []
            total = np.sum(self.weights, )
            return

        @staticmethod
        def cross_entropy_loss(output, expected):
            expected = np.array(expected)
            if len(expected.shape) == 1:
                expected = np.atleast_2d(expected).T

            total = - np.sum(np.log(output) * expected, axis=1)
            return total

        @staticmethod
        def softmax_cost(output, expected):
            return np.mean(NeuralNet.Layer.cross_entropy_loss(output, expected))

    class InputLayer(Layer):
        def __init__(self, input_count):
            super().__init__(input_count)

    class HiddenLayer(Layer):
        def __init__(self, neuron_count):
            super().__init__(neuron_count, self.ReLu, self.Relu_deriv)

    class SoftMaxOutputLayer(Layer):
        def __init__(self, neuron_count):
            super().__init__(neuron_count, self.softmax, self.softmax_deriv)

    class RegressionOutputLayer(Layer):
        def __init__(self, neuron_count):
            super().__init__(neuron_count, self.linear_activation, self.linear_deriv)

    def __init__(self, lr, layer_sizes, momentum=0.0, name="Neural Network"):
        self.lr = lr
        self.momentum = momentum
        self.network = np.empty(len(layer_sizes), dtype=self.Layer)
        self.build_network(layer_sizes)
        self.name = name

    def build_network(self, layer_sizes):
        self.network[0] = self.InputLayer(layer_sizes[0])
        for i in range(1, len(layer_sizes) - 1):
            new_layer = self.HiddenLayer(layer_sizes[i])
            new_layer.generate_weights(layer_sizes[i], layer_sizes[i-1])
            self.network[i] = new_layer
        if layer_sizes[-1] > 1:
            out_layer = self.SoftMaxOutputLayer(layer_sizes[-1])
        else:
            out_layer = self.RegressionOutputLayer(1)
        out_layer.generate_weights(layer_sizes[-1], layer_sizes[-2])
        self.network[-1] = out_layer

    def forward_prop(self, data):
        for i in range(1, len(self.network)):
            data = self.network[i].feed_forward(data)
        return data

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


net = NeuralNet(0.01, [30, 15, 2], 0.9, name="TESTING")
print(net)

