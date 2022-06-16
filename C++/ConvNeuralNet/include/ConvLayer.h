//
// Created by agils on 6/10/2022.
//

#pragma once

#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <memory.h>
#include <iomanip>
#include <random>
#include <chrono>

#define tensor vector<vector<vector<double>>>
#define matrix vector<vector<double>>

using std::vector;
using std::cout;
using std::endl;
using std::uniform_real_distribution;
using std::default_random_engine;
using std::string;
using std::max;
using std::exception;
using std::setw;


void multTensors(tensor& t1, tensor& t2);
vector<double> flattenTensor(tensor dataIn);
tensor reshapeVector(vector<double> vec, int channelCount, int r, int c);
tensor addPadding(tensor dataIn, int padWidth);
tensor delPadding(tensor dataIn, int padWidth);
void printVector(vector<double> vec);
void printMatrix(matrix vec);
void printTensor(tensor vec);
void printTensorShape(tensor vec);
matrix allocMatrix(int r, int c, double val=0.0);
tensor allocTensor(int channelCount, int dim, double value=0.0);


class OutputLayer {
public:
    OutputLayer(int numNeurons, int numInputs, double lr, double m ): numNeurons(numNeurons), numInputs(numInputs), lr(lr), m(m) {
        generateWeights();
        this->weightGradients = allocMatrix(numNeurons, numInputs);
        this->prevWeightGradients = allocMatrix(numNeurons, numInputs);
        this->biasGradients = vector<double>(numNeurons);
        this->prevBiasGradients = vector<double>(numNeurons);
        this->activatedOutputs = vector<double>(numNeurons);
        this->gradients = vector<double>(numNeurons);
    }

    vector<double> propagate(vector<double> dataIn) {
        this->inputs = std::move(dataIn);
        weightedSum();
        softmax();
        return this->activatedOutputs;
    }

    vector<double> backPropagate(vector<double> nextDeltas) {
        // calculate input deltas first and then calculate weight deltas
        softmaxDeriv(std::move(nextDeltas));
        weightDeriv();
        return this->gradients;
    }

    void update() {
        for (int n = 0; n < numNeurons; n++) {
            for (int w = 0; w < numInputs; w++) {
                this->weights[n][w] -= this->weightGradients[n][w];
                this->prevWeightGradients[n][w] = this->weightGradients[n][w];
            }
            this->biases[n] -= this->biasGradients[n];
            this->prevBiasGradients[n] = this->biasGradients[n];
        }
    }

    int getNumNeurons() const {
        return numNeurons;
    }

    matrix getWeights() const {
        return weights;
    }

private:

    void weightedSum() {
        this->preOutputs = vector<double>(numNeurons);
        for (int n = 0; n < numNeurons; n++) {
            for (int w = 0; w < numInputs; w++) {
                this->preOutputs[n] += weights[n][w] * inputs[w];
            }
            this->preOutputs[n] += biases[n];
        }
    }

    void softmax() {
        double denom = 0;
        for (int n = 0; n < numNeurons; n++) {
            denom += exp(this->preOutputs[n]);
        }
        for (int n = 0; n < numNeurons; n++) {
            this->activatedOutputs[n] = exp(this->preOutputs[n]) / denom;
        }
    }

    void softmaxDeriv(vector<double> nextDeltas) {
        for (int n = 0; n < numNeurons; n++) {
            this->gradients[n] = this->activatedOutputs[n] - nextDeltas[n];
        }
    }

    void weightDeriv() {
        for (int n = 0; n < numNeurons; n++) {
            for (int w = 0; w < numInputs; w++) {
                this->weightGradients[n][w] = this->inputs[w] * this->gradients[n] * this->lr + this->prevWeightGradients[n][w] * this->m;
            }
            this->biasGradients[n] = this->gradients[n] * this->lr + this->prevBiasGradients[n] * this->m;
        }
    }

    void generateWeights() {
        double weightLimit = (sqrt(6.0) / sqrt(numInputs + numNeurons));
        weights = allocRandMatrix(numNeurons, numInputs, -weightLimit, weightLimit);
        biases = allocRandVector(numNeurons, -weightLimit, weightLimit);
    }

    vector<double> allocRandVector(int n, double lower, double upper) {
        unif = uniform_real_distribution<double>(lower, upper);
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        vector<double> res(n);
        for (int i = 0; i < n; i++) {
            res[i] = unif(rng);
        }
        return res;
    }

    matrix allocRandMatrix(int r, int c, double lower, double upper) {
        unif = uniform_real_distribution<double>(lower, upper);
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        matrix res(r);
        for (int row = 0; row < r; row++) {
            vector<double> vec(c);
            for (int col = 0; col < c; col++) {
                vec[col] = unif(rng);
            }
            res[row] = vec;
        }
        return res;
    }

    matrix weights;
    matrix weightGradients;
    matrix prevWeightGradients;
    vector<double> biases;
    vector<double> biasGradients;
    vector<double> prevBiasGradients;
    vector<double> preOutputs;
    vector<double> activatedOutputs;
    vector<double> gradients;
    vector<double> inputs;
    int numNeurons;
    int numInputs;
    double lr;
    double m;
    uniform_real_distribution<double> unif;
    default_random_engine rng;
};


class HiddenLayer {
public:
    HiddenLayer(int numNeurons, int numInputs, double lr, double m): numNeurons(numNeurons), numInputs(numInputs), lr(lr), m(m) {
        generateWeights();
        this->weightGradients = allocMatrix(numNeurons, numInputs);
        this->prevWeightGradients = allocMatrix(numNeurons, numInputs);
        this->biasGradients = vector<double>(numNeurons);
        this->prevBiasGradients = vector<double>(numNeurons);
        this->activatedOutputs = vector<double>(numNeurons);
        this->gradients = vector<double>(numNeurons);
    }

    vector<double> propagate(vector<double> dataIn) {
        this->inputs = dataIn;
        weightedSum();
        ReLu();
        return this->activatedOutputs;
    }

    vector<double> backPropagate(vector<double> nextDeltas, matrix nextWeights) {
        // calculate input deltas first and then calculate weight deltas
        ReLuDeriv(std::move(nextDeltas), std::move(nextWeights));
        weightDeriv();
        return this->gradients;
    }

    void update() {
        for (int n = 0; n < numNeurons; n++) {
            for (int w = 0; w < numInputs; w++) {
                this->weights[n][w] -= this->weightGradients[n][w];
                this->prevWeightGradients[n][w] = this->weightGradients[n][w];
            }
            this->biases[n] -= this->biasGradients[n];
            this->prevBiasGradients[n] = this->biasGradients[n];
        }
    }

    int getNumWeights() const {
        return numInputs;
    }

    int getNumNeurons() const {
        return numNeurons;
    }

    matrix getWeights() const {
        return weights;
    }

private:

    void weightedSum() {
        this->preOutputs = vector<double>(numNeurons);
        for (int n = 0; n < numNeurons; n++) {
            for (int w = 0; w < numInputs; w++) {
                this->preOutputs[n] += weights[n][w] * inputs[w];
            }
            this->preOutputs[n] += biases[n];
        }
    }

    void ReLu() {
        for (int n = 0; n < numNeurons; n++) {
            if (preOutputs[n] > 0) {
                this->activatedOutputs[n] = preOutputs[n];
            }
            else {
                this->activatedOutputs[n] = 0.0;
            }
        }
    }

    void ReLuDeriv(vector<double> nextDeltas, matrix nextWeights) {
        for (int n = 0; n < numNeurons; n++) {
            if (preOutputs[n] > 0) {
                double total = 0;
                for (int nextN = 0; nextN < nextWeights.size(); nextN++) {
                    total += nextWeights[nextN][n] * nextDeltas[nextN];
                }
                this->gradients[n] = total;
            }
            else {
                this->gradients[n] = 0;
            }
        }
    }

    void weightDeriv() {
        for (int n = 0; n < numNeurons; n++) {
            for (int w = 0; w < numInputs; w++) {
                this->weightGradients[n][w] = this->inputs[w] * this->gradients[n] * this->lr + this->prevWeightGradients[n][w] * this->m;
            }
            this->biasGradients[n] = this->gradients[n] * this->lr + this->prevBiasGradients[n] * this->m;
        }
    }

    void generateWeights() {
        double weightLimit = (sqrt(6.0) / sqrt(numInputs + numNeurons));
        weights = allocRandMatrix(numNeurons, numInputs, -weightLimit, weightLimit);
        biases = allocRandVector(numNeurons, -weightLimit, weightLimit);
    }



    vector<double> allocRandVector(int n, double lower, double upper) {
        unif = uniform_real_distribution<double>(lower, upper);
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        vector<double> res(n);
        for (int i = 0; i < n; i++) {
            res[i] = unif(rng);
        }
        return res;
    }

    matrix allocRandMatrix(int r, int c, double lower, double upper) {
        unif = uniform_real_distribution<double>(lower, upper);
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        matrix res(r);
        for (int row = 0; row < r; row++) {
            vector<double> vec(c);
            for (int col = 0; col < c; col++) {
                vec[col] = unif(rng);
            }
            res[row] = vec;
        }
        return res;
    }

    matrix weights;
    matrix weightGradients;
    matrix prevWeightGradients;
    vector<double> biases;
    vector<double> biasGradients;
    vector<double> prevBiasGradients;
    vector<double> preOutputs;
    vector<double> activatedOutputs;
    vector<double> gradients;
    vector<double> inputs;
    int numNeurons;
    int numInputs;
    double lr;
    double m;
    uniform_real_distribution<double> unif;
    default_random_engine rng;
};


class KernelLayer {
public:
    // constructor
    KernelLayer(int numChannels, int kernelSize, int inputChannels, int inputDim, double lr, double m, bool padding, bool pooling, string poolType): numChannels(numChannels), kernelSize(kernelSize), inputChannels(inputChannels), inputDim(inputDim), lr(lr), m(m), padding(padding), pooling(pooling), poolType(std::move(poolType)) {
        // set up rng
        this->unif = uniform_real_distribution<double>(-1, 1);
        this->rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        // set pad width for kernel input padding
        if (padding) {
            this->padWidth = ceil(double(kernelSize) / 2);
        }
        // calc output dimensions of kernel
        this->kernelOutputDim = (inputDim + 2 * padWidth - kernelSize) + 1;
        // re-calc output if pooling enabled
        if (pooling) {
            // if output dimension divisible by 2
            if (this->kernelOutputDim % 2 != 0) {
                this->outputDim = int((this->kernelOutputDim + 1) / 2);
            }
            // else
            else {
                this->outputDim = int(this->kernelOutputDim / 2);
            }
            // allocate space for pool indexing tensor
            poolIndices = allocPoolTensor(numChannels, this->outputDim);
            this->outputs = allocTensor(numChannels, this->outputDim);
        }
        else {
            this->outputDim = kernelOutputDim;
        }
        this->activationDeltas = allocTensor(numChannels, this->kernelOutputDim);
        // allocate output tensor space
        this->kernelPreOutputs = allocTensor(numChannels, this->kernelOutputDim);
        this->kernelActivatedOutputs = allocTensor(numChannels, this->kernelOutputDim);
        // generate kernel values
        this->kernels = allocRandomTensor(numChannels, kernelSize);
        this->kernelGradients = allocTensor(numChannels, this->kernelSize);
        this->kernelPrevGradients = allocTensor(numChannels, this->kernelSize);
    }

    // propagation method
    tensor propagate(tensor dataIn) {
        // adjust input with padding if enabled
        if (this->padding) {
            dataIn = addPadding(dataIn, padWidth);
        }
        this->inputs = dataIn;
        this->inputDim = dataIn[0].size();
        // for every channel
        for (int channelIndex = 0; channelIndex < this->numChannels; channelIndex++) {
            // for every kernel output row
            for (int row = 0; row < this->kernelOutputDim; row++) {
                // for every kernel output column
                for (int col = 0; col < this->kernelOutputDim; col++) {
                    double kernelSum = 0;
                    for (int kRow = 0;  kRow < this->kernelSize; kRow++) {
                        for (int kCol = 0; kCol < this->kernelSize; kCol++) {
                            for (int i = 0; i < inputChannels; i++) {
                                kernelSum += dataIn[i][row + kRow][col + kCol] * this->kernels[channelIndex][kRow][kCol];
                            }
                        }
                    }
                    this->kernelPreOutputs[channelIndex][row][col] = kernelSum;
                }
            }
        }
        this->ReLu();
        if (!pooling) {
            this->outputs = kernelActivatedOutputs;
        }
        else {
            this->pool(kernelActivatedOutputs);
        }
        return this->outputs;
    }

    tensor backPropagate(const tensor& deltas) {
        // if pooling layer present
        if (pooling) {
            poolDeriv(deltas);
        }
        // if no pooling layers
        else {
            this->poolDeltas = deltas;
        }

        // calculate activation function delta
        ReLuDeriv();
        // chain rule with previous delta
        multTensors(this->activationDeltas, this->poolDeltas);
        // calculate input derivatives
        inputDeriv();
        // calculate kernel derivatives and apply gradients
        kernelDeriv();
        updateKernels();
        return this->inputDeltas;
    }

    void update() {
        for (int channelIndex = 0; channelIndex < numChannels; channelIndex++) {
            for (int row = 0; row < kernelSize; row++) {
                for (int col = 0; col < kernelSize; col++) {
                    auto res = this->kernelGradients[channelIndex][row][col] * lr + this->kernelPrevGradients[channelIndex][row][col] * m;
                    this->kernels[channelIndex][row][col] -= res;
                    this->kernelPrevGradients[channelIndex][row][col] = res;
                }
            }
        }
    }

    int getPadWidth() const {
        return this->padWidth;
    }

    int getNumChannels() const {
        return this->numChannels;
    }

    int getInputChannels() const {
        return this->inputChannels;
    }

    int getKernelDim() const {
        return this->kernelSize;
    }

    int getOutputDim() const {
        return this->outputDim;
    }

private:

    void updateKernels() {
        for (int channelIndex = 0; channelIndex < numChannels; channelIndex++) {
            for (int row = 0; row < kernelSize; row++) {
                for (int col = 0; col < kernelSize; col++) {
                    this->kernelGradients[channelIndex][row][col] = this->kernelDeltas[channelIndex][row][col];
                }
            }
        }
    }

    void kernelDeriv() {
        this->kernelDeltas = allocTensor(numChannels, this->kernelSize);
        for (int channelIndex = 0; channelIndex < numChannels; channelIndex++) {
            for (int row = 0; row < kernelOutputDim; row++) {
                for (int col = 0; col < kernelOutputDim; col++) {
                    // if deltas are non-zero
                    if (this->activationDeltas[channelIndex][row][col] != 0) {
                        // add all values in the kernel range
                        for (int kRow = 0; kRow < kernelSize; kRow++) {
                            for (int kCol = 0; kCol < kernelSize; kCol++) {
                                for (int i = 0; i < inputChannels; i++) {
                                    this->kernelDeltas[channelIndex][kRow][kCol] += this->activationDeltas[channelIndex][row][col] * this->inputs[i][row + kRow][col + kCol];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void inputDeriv() {
        this->inputDeltas = allocTensor(inputChannels, inputDim);
        for (int channelIndex = 0; channelIndex < numChannels; channelIndex++) {
            for (int row = 0; row < kernelOutputDim; row++) {
                for (int col = 0; col < kernelOutputDim; col++) {
                    if (this->activationDeltas[channelIndex][row][col] != 0) {
                        for (int kRow = 0; kRow < kernelSize; kRow++) {
                            for (int kCol = 0; kCol < kernelSize; kCol++) {
                                for (int i = 0; i < inputChannels; i++) {
                                    this->inputDeltas[i][row + kRow][col + kCol] += this->kernels[channelIndex][kRow][kCol] * this->activationDeltas[channelIndex][row][col];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void ReLu() {
        for (int channelIndex = 0; channelIndex < this->numChannels; channelIndex++) {
            for (int row = 0; row < this->kernelOutputDim; row++) {
                for (int col = 0; col < this->kernelOutputDim; col++) {
                    this->kernelActivatedOutputs[channelIndex][row][col] = max({0.0, this->kernelPreOutputs[channelIndex][row][col]});
                }
            }
        }
    }

    void ReLuDeriv() {
        for (int channelIndex = 0; channelIndex < this->numChannels; channelIndex++) {
            for (int row = 0; row < this->kernelOutputDim; row++) {
                for (int col = 0; col < this->kernelOutputDim; col++) {
                    if (this->kernelPreOutputs[channelIndex][row][col] > 0) {
                        this->activationDeltas[channelIndex][row][col] = 1;
                    }
                    else {
                        this->activationDeltas[channelIndex][row][col] = 0;
                    }
                }
            }
        }
    }

    void pool(tensor dataIn) {
        // adjust dimensions of input data for pooling
        int d = this->kernelOutputDim;
        if (d % 2 != 0) {
            vector<double> zeroVec(d + 1, 0.0);
            for (int channelIndex = 0; channelIndex < numChannels; channelIndex++) {
                for (int row = 0; row < d; row++) {
                    dataIn[channelIndex][row].push_back(0.0);
                }
                dataIn[channelIndex].push_back(zeroVec);
            }
        }

        // if pooling type is max pooling
        if (this->poolType == "max") {
            // for every channel in pool output
            for (int channelIndex = 0; channelIndex < this->numChannels; channelIndex++) {
                // for every row in pool output
                for (int row = 0; row < this->outputDim; row++) {
                    // for every col in pool output
                    for (int col = 0; col < this->outputDim; col++) {
                        bool varAssigned = false;
                        vector<int> maxIndices = {0, 0};
                        double maxVal = 0;
                        for (int fieldRow = 0; fieldRow < 2; fieldRow++) {
                            for (int fieldCol = 0; fieldCol < 2; fieldCol++) {
                                int rowIndex = row * 2 + fieldRow;
                                int colIndex = col * 2 + fieldCol;
                                double val = dataIn[channelIndex][rowIndex][colIndex];
                                if (val > maxVal || !varAssigned) {
                                    maxVal = val;
                                    maxIndices = {rowIndex, colIndex};
                                    varAssigned = true;
                                }
                            }
                        }
                        this->poolIndices[channelIndex][row][col] = maxIndices;
                        this->outputs[channelIndex][row][col] = maxVal;
                    }
                }
            }
        }
        else if (this->poolType == "avg") {
            // for every channel in pool output
            for (int channelIndex = 0; channelIndex < this->numChannels; channelIndex++) {
                // for every row in pool output
                for (int row = 0; row < this->outputDim; row++) {
                    // for every col in pool output
                    for (int col = 0; col < this->outputDim; col++) {
                        double total = 0;
                        for (int fieldRow = 0; fieldRow < 2; fieldRow++) {
                            for (int fieldCol = 0; fieldCol < 2; fieldCol++) {
                                double val = dataIn[channelIndex][row * 2 + fieldRow][col * 2 + fieldCol];
                                total += val;
                            }
                        }
                        this->outputs[channelIndex][row][col] = total / 4.0;
                    }
                }
            }
        }
    }

    void poolDeriv(const tensor& deltas) {
        // if pool type is max-pooling
        if (poolType == "max") {
            this->poolDeltas = allocTensor(numChannels, this->kernelOutputDim);
            for (int channelIndex = 0; channelIndex < this->numChannels; channelIndex++) {
                for (int row = 0; row < this->outputDim; row++) {
                    for (int col = 0; col < this->outputDim; col++) {
                        vector<int> curPoolIndex = this->poolIndices[channelIndex][row][col];
                        this->poolDeltas[channelIndex][curPoolIndex[0]][curPoolIndex[1]] = deltas[channelIndex][row][col];
                    }
                }
            }
        }
            // if pool type is average pooling
        else if (poolType == "avg") {
            this->poolDeltas = allocTensor(numChannels, this->kernelOutputDim, 0.25);
            for (int channelIndex = 0; channelIndex < this->numChannels; channelIndex++) {
                for (int row = 0; row < this->kernelOutputDim; row++) {
                    for (int col = 0; col < this->kernelOutputDim; col++) {
                        this->poolDeltas[channelIndex][row][col] *= deltas[channelIndex][int(row / 2)][int(col / 2)];
                    }
                }
            }
        }
    }

    // creates a random tensor of the desired shape
    tensor allocRandomTensor(int channelCount, int dim) {
        tensor resTensor(channelCount);
        for (auto & channelIndex : resTensor) {
            matrix curChannel(dim);
            for (auto & row : curChannel) {
                vector<double> curRow(dim);
                for (double & col : curRow) {
                    col = unif(rng);
                }
                row = curRow;
            }
            channelIndex = curChannel;
        }
        return resTensor;
    }

    // creates an empty tensor for pool indexing
    vector<vector<vector<vector<int>>>> allocPoolTensor(int channelCount, int dim) {
        vector<vector<vector<vector<int>>>> resTensor(channelCount);
        for (auto & channelIndex : resTensor) {
            vector<vector<vector<int>>> curChannel(dim);
            for (auto & row : curChannel) {
                vector<vector<int>> curRow(dim);
                for (auto & col : curRow) {
                    vector<int> curCol(2);
                    col = curCol;
                }
                row = curRow;
            }
            channelIndex = curChannel;
        }
        return resTensor;
    }

    tensor kernels;
    tensor inputs;
    tensor inputDeltas;
    tensor poolDeltas;
    tensor activationDeltas;
    tensor kernelDeltas;
    tensor kernelGradients;
    tensor kernelPrevGradients;
    tensor kernelPreOutputs;
    tensor kernelActivatedOutputs;
    tensor outputs;
    vector<vector<vector<vector<int>>>> poolIndices;
    uniform_real_distribution<double> unif;
    default_random_engine rng;
    int kernelSize;
    int numChannels;
    int inputDim;
    int inputChannels;
    int kernelOutputDim;
    int outputDim;
    double lr;
    double m;
    bool padding;
    int padWidth = 0;
    bool pooling;
    string poolType;
};


void multTensors(tensor& t1, tensor& t2) {
    for (int channelIndex = 0; channelIndex < t1.size(); channelIndex++) {
        for (int row = 0; row < t1[channelIndex].size(); row++) {
            for (int col = 0; col < t1[channelIndex][row].size(); col++) {
                t1[channelIndex][row][col] *= t2[channelIndex][row][col];
            }
        }
    }
}

vector<double> flattenTensor(tensor dataIn) {
    vector<double> res(dataIn.size() * dataIn[0].size() * dataIn[0][0].size());

    for (int channelIndex = 0; channelIndex < dataIn.size(); channelIndex++) {
        for (int row = 0; row < dataIn[channelIndex].size(); row++) {
            for (int col = 0; col < dataIn[channelIndex][row].size(); col++) {
                int index = row * dataIn.size() * dataIn[0][0].size() + (col * dataIn.size()) + channelIndex;
                res[index] = dataIn[channelIndex][row][col];
            }
        }
    }
    return res;
}

tensor reshapeVector(vector<double> vec, int channelCount, int r, int c) {
    if (channelCount * r * c != vec.size()) {
        throw exception("Invalid sizes");
    }

    tensor res = allocTensor(channelCount, r, c);
    int indexCount = 0;

    for (int channelIndex = 0; channelIndex < channelCount; channelIndex++) {
        for (int row = 0; row < r; row++) {
            for (int col = 0; col < c; col++) {
                res[channelIndex][row][col] = vec[indexCount];
                indexCount++;
            }
        }
    }
    return res;
}

tensor addPadding(tensor dataIn, int padWidth) {
    // dimension of padded output
    int newDimension = dataIn[0].size() + 2 * padWidth;
    // number of channels
    int numChannels = dataIn.size();
    // padded result tensor
    tensor paddedRes(numChannels);
    // zero vector for padding
    vector<double> zeroVector(newDimension, 0.0);
    // for every channel
    for (int channelIndex = 0; channelIndex < dataIn.size(); channelIndex++) {
        // new channel to be added to padded result tensor
        matrix newChannel(newDimension);
        // add top padding
        for (int padIndex = 0; padIndex < padWidth; padIndex++) {
            newChannel[padIndex] = zeroVector;
        }
        // add side padding
        for (int row = 0; row < dataIn[channelIndex].size(); row++) {
            // current row to be added to current channel
            vector<double> currentRow(newDimension, 0.0);
            for (int col = 0; col < dataIn[channelIndex][row].size(); col++) {
                // update current row column values
                currentRow[col + padWidth] = dataIn[channelIndex][row][col];
            }
            // update new channel row
            newChannel[row + padWidth] = currentRow;
        }
        // add bottom padding
        for (int padIndex = newDimension - padWidth; padIndex < newDimension; padIndex++) {
            newChannel[padIndex] = zeroVector;
        }
        // update padded result channel
        paddedRes[channelIndex] = newChannel;
    }
    // return padded result
    return paddedRes;
}

tensor delPadding(tensor dataIn, int padWidth) {
    // dimension of output with padding removed
    int newDimension = dataIn[0].size() - 2 * padWidth;
    // number of channels
    int numChannels = dataIn.size();
    // result tensor with padding removed
    tensor unpaddedRes(numChannels);
    // for every channel
    for (int channelIndex = 0; channelIndex < dataIn.size(); channelIndex++) {
        // current channel
        matrix newChannel(newDimension);
        // add side padding
        for (int row = padWidth; row < dataIn[channelIndex].size() - padWidth; row++) {
            // current row
            vector<double> currentRow(newDimension);
            // update columns in current row
            for (int col = padWidth; col < dataIn[channelIndex][row].size() - padWidth; col++) {
                currentRow[col - padWidth] = dataIn[channelIndex][row][col];
            }
            // update current channel row
            newChannel[row - padWidth] = (currentRow);
        }
        // update result tensor with channel
        unpaddedRes[channelIndex] = newChannel;
    }
    // return result tensor
    return unpaddedRes;
}

template <typename T>
void printVector(vector<T> vec) {
    cout << "[";
    for (int i = 0; i < vec.size(); i++) {
        cout << setw(10) <<vec[i];
        if (i != vec.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;
}

void printMatrix(matrix vec) {
    cout << "[" << endl;
    for (auto & i : vec) {
        cout << "   [ ";
        for (int m = 0; m < i.size(); m++) {
            cout << setw(10) <<i[m];
            if (m != i.size() - 1) {
                cout << ", ";
            }
        }
        cout << "]" << endl;
    }
    cout << "]" << endl;
}

void printTensor(tensor vec) {
    cout << "[" << endl;
    for (auto & n : vec) {
        cout << "   [" << endl;
        for (auto & i : n) {
            cout << "       [ ";
            for (int m = 0; m < i.size(); m++) {
                cout << setw(10) << i[m];
                if (m != i.size() - 1) {
                    cout << ", ";
                }
            }
            cout << "]" << endl;
        }
        cout << "   ]" << endl;
    }
    cout << "]" << endl;
}

void printTensorShape(tensor vec) {
    cout << "[" << vec.size() << ", " << vec[0].size() << ", " << vec[0][0].size() << "]" << endl;
}

matrix allocMatrix(int r, int c, double val) {
    matrix res(r);
    for (int row = 0; row < r; row++) {
        res[row] = vector<double>(c, val);
    }
    return res;
}

// creates an empty tensor of the desired shape
tensor allocTensor(int channelCount, int dim, double value) {
    tensor resTensor(channelCount);
    for (auto & channelIndex : resTensor) {
        matrix curChannel(dim);
        for (auto & row : curChannel) {
            vector<double> curRow(dim, value);
            row = curRow;
        }
        channelIndex = curChannel;
    }
    return resTensor;
}


void normalize(vector<tensor>& dataIn, vector<double> minMaxRanges) {
    // for data point channel
    for (int d = 0; d < dataIn.size(); d++) {
        for (int channelIndex = 0; channelIndex < dataIn[d].size(); channelIndex++) {
            // for every data trait
            double min = minMaxRanges[0];
            double max = minMaxRanges[1];
            for (int row = 0; row < dataIn[d][channelIndex].size(); row++) {
                // gets the min and max values for the data range
                // for every data point x in the dataset
                for (int col = 0; col < dataIn[d][channelIndex][row].size(); col++) {
                    dataIn[d][channelIndex][row][col] = (dataIn[d][channelIndex][row][col] - min) / (max - min);
                }
            }
        }
    }
}

