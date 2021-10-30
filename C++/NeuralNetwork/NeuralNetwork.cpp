//
// Created by Alex Gilsoul on 10/19/2021.
//

#include "NeuralNetwork.h"


//Public Methods

double NeuralNetwork::Neuron::calculate(vector<double> input) {
    prevInputs = input;
    double total = 0;

    for (int i = 0; i < weights.size(); i++) {
        total += weights[i] * input[i];
    }
    total += bias;
    return total;
}

NeuralNetwork::NeuralNetwork(int numLayers, vector<int> neurons, double learningRate) {
    this->learningRate = learningRate;
    for (int i = 0; i < numLayers; i++) {
        vector<Neuron*> tempLayer;
        for (int n = 0; n < neurons[i]; n++) {
            Neuron* tempNeuron = new Neuron;
            if (i != 0) {
                initializeWeights(neurons[i-1], tempNeuron);
            }
            tempLayer.push_back(tempNeuron);
        }
        layers.push_back(tempLayer);
    }

    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};


    rng.seed(ss);
    //rng.seed(5);
}


void NeuralNetwork::normalize(vector<vector<double>>& input) {
    for (int p = 0; p < input[0].size(); p++) {
        vector<double> curData;
        for (auto & i : input) {
            curData.push_back(i[p]);
        }
        auto sortedData = sortVector(curData);
        for (auto & i : input) {
            i[p] = (i[p] - sortedData[0]) / (sortedData[sortedData.size() - 1] - sortedData[0]);
        }
    }
}

//still working on this
void NeuralNetwork::train(vector<vector<double>> input, vector<vector<double>> allResults, int iterations) {
    //initialize input neuron weights
    for (int i = 0; i < layers[0].size(); i++) {
        initializeWeights(input.size(), layers[0][i]);
    }
    //for every iteration
    vector<vector<double>> weightChanges;
    for (int i = 0; i < iterations; i++) {
        cout << i << endl;
        for (int x = 0; x < input.size(); x++) {
            //printVector(forwardProp(input[0]));

            auto desiredResult = allResults[x];
            //cout << "data point #" << x << endl;
            //gets final result of forward propogation
            vector<double> finalResult = forwardProp(input[x]);
            //first layer backprop
            vector<double> nextDeltas;
            vector<double> firstChange;
            for (int neuronCount = 0; neuronCount < layers[layers.size() - 1].size();neuronCount++) {
                auto curN = layers[layers.size()- 1][neuronCount];
                curN->delta = finalGradient(curN, desiredResult[neuronCount]);
                nextDeltas.push_back(curN->delta);
            }
            //hidden layer backprop
            for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                //for every neuron in the layer
                vector<double> tempDeltas;
                for (int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    curN->delta = hiddenGradient(curN, neuronCount, layers[layerCount + 1], nextDeltas);
                    tempDeltas.push_back(curN->delta);
                    //for every weight
                }
                nextDeltas = tempDeltas;
            }

            //updating weights
            for (int layerCount = layers.size() - 1; layerCount >= 0; layerCount--) {
                for (int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    for (int w = 0; w < curN->weights.size(); w++) {
                        curN->weights[w] -= weightDerivative(curN->delta, curN->prevInputs[w]) * learningRate;
                       // cout << curN->weights[w] << " ";
                    }
                    //cout << endl;
                    curN->bias -= curN->delta * learningRate;
                }
            }
        }
    }
}




vector<double> NeuralNetwork::forwardProp(vector<double> input) {
    auto data = input;

    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        vector<double> layerResults;
        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].size(); neuronIndex++) {
            auto tempNPointer = layers[layerIndex][neuronIndex];
            double neuronResult = tempNPointer->calculate(data);
            tempNPointer->prevInputs = data;
            tempNPointer->output = neuronResult;
            //cout << neuronResult << " s: ";
            layerResults.push_back(sigmoid(neuronResult));
            //cout << sigmoid(neuronResult) << " ";
        }
        //cout << endl << "NEW LAYER" << endl;
        data = layerResults;
    }
    //cout << "****************NEW FORWARDPROP*****************" << endl;
    //cout << "nan weights: " << endl;
    //printVector(layers[0][7]->weights);
    //cout << endl << endl;
    //cout << endl;
    return data;
}


void NeuralNetwork::printVector(vector<double> input) {
    cout << "{ " << input[0];
    for (int i = 1; i < input.size(); i++) {
        cout << ", " << input[i];
    }
    cout << " }";
}


vector<vector<double>> NeuralNetwork::vectorSplit(vector<vector<double>> vec, int start, int fin) {
    vector<vector<double>> newVec;
    for (int i = start; i <= fin; i++) {
        newVec.push_back(vec[i]);
    }
    return newVec;
}

double NeuralNetwork::test(vector<vector<double>>& testData, vector<vector<double>>& testLabel) {
    double accuracy = 0;
    for (int i = 0; i < testData.size(); i++) {
        //printVector(testLabel[i]);
        vector<double> tempResult = forwardProp(testData[i]);
        double newResult;
        if (tempResult[0] > 0.5) {
            newResult = 1;
        }
        else {
            newResult = 0;
        }
        cout << "result: " << newResult << " actual: ";
        printVector(testLabel[i]);
        cout << endl <<endl;
        if (newResult == testLabel[i][0]) {
            accuracy++;
        }
    }
    return accuracy / testData.size() * 100;
}


//Private Methods
double NeuralNetwork::sigmoid(double input) {
    return 1 / (1 + exp(-input));
}

double NeuralNetwork::sigmoidDeriv(double input) {
    return sigmoid(input) * (1 - sigmoid(input));
}

void NeuralNetwork::initializeWeights(int numWeights, Neuron* newN) {
    std::uniform_real_distribution<double> unif(-1, 1);
    for (int i = 0; i < numWeights; i++) {
        newN->weights.push_back(unif(rng));
        //newN->weights.push_back(sigmoid(i));
    }
}

//FINISH THESE
double NeuralNetwork::finalGradient(Neuron* curN, double expected) {
    return sigmoidDeriv(curN->output) * (sigmoid(curN->output) - expected);
}

double NeuralNetwork::hiddenGradient(Neuron* curN, int nIndex, vector<Neuron*> nextLayer, vector<double> nextDeltas) {
    double total = 0;
    for (int i = 0; i < nextLayer.size(); i++) {
        auto newN = nextLayer[i];
        total += newN->weights[nIndex] * nextDeltas[i];
    }
    return sigmoidDeriv(curN->output) * total;
}

double NeuralNetwork::weightDerivative(double neuronError, double prevNeuron) {
    return neuronError * prevNeuron;
}

vector<double> NeuralNetwork::sortVector(vector<double> vec) {
    vector<double> sortedData;
    sortedData.push_back(vec[0]);
    for (int x = 1; x < vec.size(); x++) {
        for (int y = 0; y < sortedData.size(); y++) {
            if (vec[x] < sortedData[y]) {
                sortedData.insert(sortedData.begin() + y, vec[x]);
                break;
            }
            else if (y == sortedData.size() - 1) {
                sortedData.push_back(vec[x]);
                break;
            }
        }
    }
    return sortedData;
}


