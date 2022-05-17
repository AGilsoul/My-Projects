#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <random>
#include <string>

__global__ void neuronSum(int, int, double*, double*, double*);

__global__ void layerWeightedSumCUDA(int, int, double*, double*, double*);

__global__ void softmaxCUDA(int, double, double*, double*);

__global__ void softmaxDerivCUDA(int, double*, double*, double*);

__global__ void reluCUDA(double* , double* , int );

__global__ void reluDerivCUDA(int , double* , double* );

__global__ void hiddenGradient(int , int , int , double* , double* , double* );

__global__ void updateLayerWeights(int, int, double, double, double*, double*, double*, double*);

__global__ void updateLayerBiases(int, double, double, double*, double*, double*);

__global__ void dataNormalize(int , int , double* , double* , double* );

int readMnist(std::vector<double>&, std::vector<double>&);

class netGPU {
public:

    // constructor
    netGPU(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, double lr=0.01, double m=0.0) {
        this->hiddenLayerSize = hiddenLayerSize;
        this->outputLayerSize = outputLayerSize;
        this->inputLayerSize = inputLayerSize;
        this->lr = lr;
        this->m = m;
        printf("allocating memory...\n");
        allocateMemory();
        printf("allocated memory\n");

        std::random_device dev;
        std::mt19937_64 rng(dev());
        std::uniform_real_distribution<double> unif(-1 * sqrt(6.0) / sqrt(inputLayerSize + hiddenLayerSize), sqrt(6.0) / sqrt(inputLayerSize + hiddenLayerSize));
        // hidden layer
        for (int curN = 0; curN < hiddenLayerSize; curN++) {
            for (int numWeights = 0; numWeights < inputLayerSize; numWeights++) {
                double value;
                value = unif(rng);
                hiddenWeights[curN * inputLayerSize + numWeights] = value;
                hiddenPreRes[curN * inputLayerSize + numWeights] = 0;
                hiddenWeightGradients[curN * inputLayerSize + numWeights] = 0;
            }
            hiddenBiasGradients[curN] = 0;
            hiddenDeltas[curN] = 0;
            hiddenBiases[curN] = 0;
        }
        cudaMemcpy(hiddenWeightGradientsGPU, hiddenWeightGradients, hiddenMemSize * inputLayerSize, cudaMemcpyHostToDevice);
        cudaMemcpy(hiddenBiasGradientsGPU, hiddenBiasGradients, hiddenMemSize, cudaMemcpyHostToDevice);
        cudaMemcpy(hiddenWeightsGPU, hiddenWeights, inputLayerSize * hiddenMemSize, cudaMemcpyHostToDevice);
        cudaMemcpy(hiddenBiasesGPU, hiddenBiases, hiddenMemSize, cudaMemcpyHostToDevice);
        // output layer
        for (int curN = 0; curN < outputLayerSize; curN++) {
            for (int numWeights = 0; numWeights < hiddenLayerSize; numWeights++) {
                double value;
                value = unif(rng);
                outputWeights[curN * hiddenLayerSize + numWeights] = value;
                outputPreRes[curN * hiddenLayerSize + numWeights] = 0;
                outputWeightGradients[curN * hiddenLayerSize + numWeights] = 0;
            }
            outputBiasGradients[curN] = 0;
            outputDeltas[curN] = 0;
            outputBiases[curN] = 0;
        }
        cudaMemcpy(outputWeightGradientsGPU, outputWeightGradients, outputMemSize * hiddenLayerSize, cudaMemcpyHostToDevice);
        cudaMemcpy(outputBiasGradientsGPU, outputBiasGradients, outputMemSize, cudaMemcpyHostToDevice);
        cudaMemcpy(outputWeightsGPU, outputWeights, outputMemSize * hiddenLayerSize, cudaMemcpyHostToDevice);
        cudaMemcpy(outputBiasesGPU, outputBiases, outputMemSize, cudaMemcpyHostToDevice);

        std::vector<double> data;
        std::vector<double> targets;
        printf("reading data...\n");
        int numData = readMnist(data, targets);
        printf("done reading\n");
        double* newTrainingData = &data[0];
        double* newTrainingTargets = &targets[0];

        double numThreads = 784;
        double threadsPerBlock = 32;
        double numBlocks = ceil(numThreads / threadsPerBlock);

        double* colMinimums = (double*)malloc(784 * sizeof(double));
        double* colMaximums = (double*)malloc(784 * sizeof(double));
        for (int i = 0; i < 784; i++) {
            colMinimums[i] = 0;
            colMaximums[i] = 255;
        }

        double *d_data, *d_mins, *d_max;
        cudaMalloc((void**)&d_data, numData * 784 * sizeof(double));
        cudaMalloc((void**)&d_mins,  784 * sizeof(double));
        cudaMalloc((void**)&d_max, 784 * sizeof(double));
        cudaMemcpy(d_data, newTrainingData, 784 * numData * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mins, colMinimums, 784 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max, colMaximums, 784 * sizeof(double), cudaMemcpyHostToDevice);

        printf("normalizing data...\n");
        dataNormalize<<<numBlocks, threadsPerBlock>>>(numData, 784, d_mins, d_max, d_data);
        cudaMemcpy(newTrainingData, d_data, 784 * numData * sizeof(double), cudaMemcpyDeviceToHost);
        printf("done normalizing\n");

        printf("training on %d data points...\n", numData);
        train(10, numData, newTrainingData, newTrainingTargets);


        //cudaFree(d_data);
        //cudaFree(d_mins);
        //cudaFree(d_max);
        //free(colMaximums);
        //free(colMinimums);
        //free(newTrainingData);
        //free(newTrainingTargets);
    }

    // SGD training method
    void train(int iterations, int numDataPoints, double* trainingData, double* trainingTargets) {
        cudaMalloc((void**)&currentDataSetGPU, numDataPoints * inputLayerSize * sizeof(double));
        cudaMalloc((void**)&currentTargetsGPU, numDataPoints * outputMemSize);
        cudaMemcpy(currentDataSetGPU, trainingData, numDataPoints * inputLayerSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(currentTargetsGPU, trainingTargets, numDataPoints * outputMemSize, cudaMemcpyHostToDevice);
        for (int curIter = 0; curIter < iterations; curIter++) {
            printf("\ncur iteration: %d", curIter + 1);
            for (int dataCount = 0; dataCount < numDataPoints; dataCount++) {
                double numThreads = outputLayerSize;
                double threadsPerBlock = 32;
                double numBlocks = ceil(numThreads / threadsPerBlock);

                propagate(trainingData + dataCount * inputLayerSize);
                softmaxDerivCUDA<<<numBlocks, threadsPerBlock>>>(outputLayerSize, outputActivatedResGPU, currentTargetsGPU + (dataCount * outputLayerSize), outputDeltasGPU);

                numThreads = hiddenLayerSize;
                numBlocks = ceil(numThreads / threadsPerBlock);
                hiddenGradient<<<numBlocks, threadsPerBlock>>>(hiddenLayerSize, outputLayerSize, hiddenLayerSize, outputWeightsGPU, outputDeltasGPU, hiddenDeltasGPU);
                reluDerivCUDA<<<numBlocks, threadsPerBlock>>>(hiddenLayerSize, hiddenInputsGPU, hiddenResGPU);

                // output layer weight updates
                numThreads = outputLayerSize * hiddenLayerSize;
                numBlocks = ceil(numThreads / threadsPerBlock);
                updateLayerWeights<<<numBlocks, threadsPerBlock>>>(outputLayerSize, hiddenLayerSize, lr, m, outputWeightsGPU, outputDeltasGPU, outputWeightGradientsGPU, outputInputsGPU);
                numThreads = outputLayerSize;
                numBlocks = ceil(numThreads / threadsPerBlock);
                updateLayerBiases<<<numBlocks, threadsPerBlock>>>(outputLayerSize, lr, m, outputBiasesGPU, outputDeltasGPU, outputBiasGradientsGPU);

                // hidden layer weight updates
                numThreads = inputLayerSize * hiddenLayerSize;
                numBlocks = ceil(numThreads / threadsPerBlock);

                updateLayerWeights<<<numBlocks, threadsPerBlock>>>(hiddenLayerSize, inputLayerSize, lr, m, hiddenWeightsGPU, hiddenDeltasGPU, hiddenWeightGradientsGPU, hiddenInputsGPU);
                numThreads = hiddenLayerSize;
                numBlocks = ceil(numThreads / threadsPerBlock);
                updateLayerBiases<<<numBlocks, threadsPerBlock>>>(hiddenLayerSize, lr, m, hiddenBiasesGPU, hiddenDeltasGPU, hiddenBiasGradientsGPU);
            }
        }
        printf("\n\nDone training\n");
        printf("\nTesting...\n");
        double result = test(numDataPoints, trainingData, trainingTargets);
        printf("\nTest Results: %.2f percent accuracy", result);
    }

    // mini-batch training method, uses a different GPU thread for each training sample in a batch
    void miniBatchTrain(int iterations, int batchSize, double* trainingData, double* trainingTargets) {

    }

    // in progress
    void train(int iterations, int numDataPoints, std::vector<double> trainingData, std::vector<double> trainingTargets) {
        double* newTrainingData = &trainingData[0];
        double* newTrainingTargets = &trainingTargets[0];
        train(iterations, numDataPoints, newTrainingData, newTrainingTargets);
    }

    // in progress
    void miniBatchTrain(int iterations, int batchSize, std::vector<double> trainingData, std::vector<double> trainingTargets) {
        double* newTrainingData = &trainingData[0];
        double* newTrainingTargets = &trainingTargets[0];
        miniBatchTrain(iterations, batchSize, newTrainingData, newTrainingTargets);
    }

    // tests the model accuracy
    double test(int numDataPoints, double* testingData, double* testingTargets) {
        if (outputLayerSize > 1) {
            double numCorrect = 0;
            for (int curData = 0; curData < numDataPoints; curData++) {
                propagate(testingData + curData * inputLayerSize);
                predict(outputActivatedRes);
                bool correct = true;
                for (int i = 0; i < outputLayerSize; i++) {
                    if (outputActivatedRes[i] != testingTargets[curData * outputLayerSize + i]) {
                        correct = false;
                    }
                }
                if (correct) numCorrect += 1;
            }
            return 100 * (numCorrect / numDataPoints);
        }
    }

    // converts model results into a 1-hot vector
    void predict(double* results) {
        double maxVal;
        int maxIndex = -1;
        for (int i = 0; i < outputLayerSize; i++) {
            if (maxIndex == -1 || results[i] > maxVal) {
                maxVal = results[i];
                maxIndex = i;
            }
        }
        for (int i = 0; i < outputLayerSize; i++) {
            if (i == maxIndex) {
                results[i] = 1;
            }
            else {
                results[i] = 0;
            }
        }
    }

    // destructor, frees memory
    ~netGPU() {
        free(hiddenPreRes); free(outputPreRes);
        free(hiddenRes); free(outputRes);
        free(hiddenActivatedRes); free(outputActivatedRes);
        free(hiddenDeltas); free(outputDeltas);
        free(hiddenWeights); free(outputWeights);
        free(hiddenWeightGradients); free(outputWeightGradients);
        free(hiddenBiasGradients); free(outputBiasGradients);
        cudaFree(hiddenPreResGPU); cudaFree(outputPreResGPU);
        cudaFree(hiddenResGPU); cudaFree(outputResGPU);
        cudaFree(hiddenActivatedResGPU); cudaFree(outputActivatedResGPU);
        cudaFree(hiddenDeltasGPU); cudaFree(outputDeltasGPU);
        cudaFree(hiddenBiasesGPU); cudaFree(outputBiasesGPU);
        cudaFree(hiddenWeightsGPU); cudaFree(outputWeightsGPU);
        cudaFree(hiddenInputsGPU); cudaFree(outputInputsGPU);
        cudaFree(hiddenWeightGradientsGPU); cudaFree(outputWeightGradientsGPU);
        cudaFree(hiddenBiasGradientsGPU); cudaFree(outputBiasGradientsGPU);
    }

private:

    // forward propagation
    void propagate(double* data) {
        double numThreads = inputLayerSize * hiddenLayerSize;
        double threadsPerBlock = 32;
        double numBlocks = ceil(numThreads / threadsPerBlock);

        //hidden layer propagation
        hiddenInputs = data;

        // copy arrays to GPU
        cudaMemcpy(hiddenInputsGPU, hiddenInputs, inputLayerSize * sizeof(double), cudaMemcpyHostToDevice);
        // gets weighted inputs for each neuron weight
        layerWeightedSumCUDA<<<numBlocks, threadsPerBlock>>>(hiddenLayerSize, inputLayerSize, hiddenWeightsGPU, hiddenInputsGPU, hiddenPreResGPU);

        // reset size and threads/blocks
        numThreads = hiddenLayerSize;
        numBlocks = ceil(numThreads / threadsPerBlock);

        // gets the weighted sum for each neuron
        neuronSum<<<numBlocks, threadsPerBlock>>>(hiddenLayerSize, inputLayerSize, hiddenBiasesGPU, hiddenPreResGPU, hiddenResGPU);
        cudaMemcpy(hiddenRes, hiddenResGPU, hiddenMemSize, cudaMemcpyDeviceToHost);

        // activates hidden layer outputs
        hiddenLayerActivation();

        // output layer propagation
        outputInputsGPU = hiddenActivatedResGPU;
        numThreads = hiddenLayerSize * outputLayerSize;
        numBlocks = ceil(numThreads / threadsPerBlock);

        // copy data to GPU
        // gets weighted inputs for each neuron weight
        layerWeightedSumCUDA<<<numBlocks, threadsPerBlock>>>(outputLayerSize, hiddenLayerSize, outputWeightsGPU, outputInputsGPU, outputPreResGPU);
        // copies the results from the GPU to the CPU

        // reset size and threads/blocks
        numThreads = outputLayerSize;
        numBlocks = ceil(numThreads / threadsPerBlock);

        // copy data to GPU
        // gets the weighted sum for each neuron
        neuronSum<<<numBlocks, threadsPerBlock>>>(outputLayerSize, hiddenLayerSize, outputBiasesGPU, outputPreResGPU, outputResGPU);
        cudaMemcpy(outputRes, outputResGPU, outputMemSize, cudaMemcpyDeviceToHost);

        // activates hidden layer outputs
        outputLayerActivation();
    }

    // determines which activation function to use for output layer
    void outputLayerActivation() {
        if (outputLayerSize > 1) {
            softmax();
        }
        else {
            outputActivatedRes = outputRes;
        }
    }

    // hidden layer activation
    void hiddenLayerActivation() {
        // set number of threads to the number of neurons in the layer
        double threadCount = hiddenLayerSize;
        // set threads per block to 32
        double threadsPerBlock = 32;
        // set the number of blocks equal to thread count divided by number of threads per block
        double numBlocks = ceil(threadCount / threadsPerBlock);
        // run softmax through GPU
        reluCUDA<<<numBlocks, threadsPerBlock>>>(hiddenResGPU, hiddenActivatedResGPU, hiddenLayerSize);
    }

    // softmax activation
    void softmax() {
        // set number of threads to the number of neurons in the layer
        double numThreads = outputLayerSize;
        // set threads per block to 32
        double threadsPerBlock = 32;
        // set the number of blocks equal to thread count divided by number of threads per block
        double numBlocks = ceil(numThreads / threadsPerBlock);
        // gets memory allocation size for GPU

        // get the value of the denominator for the softmax function
        double denominator = 0;
        for (int i = 0; i < outputLayerSize; i++) {
            denominator += exp(outputRes[i]);
        }

        // run softmax through GPU
        softmaxCUDA<<<numBlocks, threadsPerBlock>>>(outputLayerSize, denominator, outputResGPU, outputActivatedResGPU);
        // copy results back to CPU
        cudaMemcpy(outputActivatedRes, outputActivatedResGPU, outputMemSize, cudaMemcpyDeviceToHost);
    }

    // allocates memory for all arrays
    void allocateMemory() {
        hiddenMemSize = hiddenLayerSize * sizeof(double);
        // hidden layer allocation
        hiddenWeights = (double*) malloc(hiddenMemSize * inputLayerSize);
        hiddenWeightGradients = (double*) malloc(hiddenMemSize * inputLayerSize);
        hiddenBiasGradients = (double*) malloc(hiddenMemSize);
        hiddenBiases = (double*) malloc(hiddenMemSize);
        hiddenDeltas = (double*) malloc(hiddenMemSize);
        hiddenPreDeltas = (double*) malloc(hiddenMemSize);
        hiddenRes = (double*)malloc(hiddenMemSize);
        hiddenPreRes = (double*)malloc(hiddenMemSize * inputLayerSize);
        hiddenActivatedRes = (double*)malloc(hiddenMemSize);
        hiddenInputs = (double*)malloc(inputLayerSize * sizeof(double));
        cudaMalloc((void**)&hiddenWeightsGPU, hiddenMemSize * inputLayerSize);
        cudaMalloc((void**)&hiddenWeightGradientsGPU, hiddenMemSize * inputLayerSize);
        cudaMalloc((void**)&hiddenBiasGradientsGPU, hiddenMemSize);
        cudaMalloc((void**)&hiddenBiasesGPU, hiddenMemSize);
        cudaMalloc((void**)&hiddenDeltasGPU, hiddenMemSize);
        cudaMalloc((void**)&hiddenPreDeltasGPU, hiddenMemSize);
        cudaMalloc((void**)&hiddenResGPU, hiddenMemSize);
        cudaMalloc((void**)&hiddenPreResGPU, hiddenMemSize * inputLayerSize);
        cudaMalloc((void**)&hiddenActivatedResGPU, hiddenMemSize);
        cudaMalloc((void**)&hiddenInputsGPU, inputLayerSize * sizeof(double));
        // output layer allocation
        outputMemSize = outputLayerSize * sizeof(double);
        outputWeights = (double*) malloc(outputMemSize * hiddenLayerSize);
        outputWeightGradients = (double*) malloc(outputMemSize * hiddenLayerSize);
        outputBiasGradients = (double*) malloc(outputMemSize);
        outputBiases = (double*) malloc(outputMemSize);
        outputDeltas = (double*) malloc(outputMemSize);
        outputPreDeltas = (double*) malloc(outputMemSize);
        outputRes = (double*)malloc(outputLayerSize * hiddenMemSize);
        outputPreRes = (double*)malloc(outputLayerSize * hiddenMemSize);
        outputActivatedRes = (double*)malloc(outputMemSize);
        outputInputs = (double*)malloc(hiddenMemSize);
        cudaMalloc((void**)&outputWeightsGPU, outputMemSize * hiddenLayerSize);
        cudaMalloc((void**)&outputWeightGradientsGPU, outputMemSize * hiddenLayerSize);
        cudaMalloc((void**)&outputBiasGradientsGPU, outputMemSize);
        cudaMalloc((void**)&outputBiasesGPU, outputMemSize);
        cudaMalloc((void**)&outputDeltasGPU, outputMemSize);
        cudaMalloc((void**)&outputPreDeltasGPU, outputMemSize);
        cudaMalloc((void**)&outputResGPU, outputLayerSize * hiddenMemSize);
        cudaMalloc((void**)&outputPreResGPU, outputLayerSize * hiddenMemSize);
        cudaMalloc((void**)&outputActivatedResGPU, outputMemSize);
        cudaMalloc((void**)&outputInputsGPU, hiddenMemSize);
    }

    double* hiddenWeights;
    double* hiddenWeightGradients;
    double* hiddenBiasGradients;
    double* hiddenBiases;
    double* hiddenDeltas;
    double* hiddenPreDeltas;
    double* hiddenInputs;
    double* hiddenWeightsGPU;
    double* hiddenWeightGradientsGPU;
    double* hiddenBiasGradientsGPU;
    double* hiddenBiasesGPU;
    double* hiddenDeltasGPU;
    double* hiddenPreDeltasGPU;
    double* hiddenInputsGPU;
    double* hiddenPreRes;
    double* hiddenPreResGPU;
    double* hiddenRes;
    double* hiddenResGPU;
    double* hiddenActivatedRes;
    double* hiddenActivatedResGPU;
    double* outputWeights;
    double* outputWeightGradients;
    double* outputBiasGradients;
    double* outputBiases;
    double* outputDeltas;
    double* outputPreDeltas;
    double* outputInputs;
    double* outputWeightsGPU;
    double* outputWeightGradientsGPU;
    double* outputBiasGradientsGPU;
    double* outputBiasesGPU;
    double* outputDeltasGPU;
    double* outputPreDeltasGPU;
    double* outputInputsGPU;
    double* outputPreRes;
    double* outputPreResGPU;
    double* outputRes;
    double* outputResGPU;
    double* outputActivatedRes;
    double* outputActivatedResGPU;
    double* currentDataSetGPU;
    double* currentTargetsGPU;
    double lr;
    double m;
    int outputLayerSize;
    int inputLayerSize;
    int hiddenLayerSize;
    size_t hiddenMemSize;
    size_t outputMemSize;
};


int main() {
    netGPU myTest(784, 32, 10, 0.001, 0.5);
    return 0;
}


// reads data from MNIST file
int readMnist(std::vector<double>& data, std::vector<double>& target) {
    std::string sList[785];
    for (unsigned int i = 0; i < 785; i++) {
        std::string temp = "";
        sList[i] = temp;
    }
    std::ifstream fin("../res/mnist_train.csv", std::ios::in);
    int dataCount = 0;
    int listSize = sizeof(sList) / sizeof(sList[0]);
    while (dataCount < 36000) {
        for (unsigned int i = 0; i < listSize; i++) {
            if (i != listSize - 1) {
                std::getline(fin, sList[i], ',');
            }
            else {
                getline(fin, sList[i], '\n');
            }
            if (i == 0) {
                for (int x = 0; x < 10; x++) {
                    if (stod(sList[i]) == x) {
                        target.push_back(1);
                    }
                    else {
                        target.push_back(0);
                    }
                }
                dataCount++;
            }
            else {
                data.push_back(stod(sList[i]));
            }
        }
    }
    fin.close();
    return dataCount;
}

// gets the total weighted sum for each neuron
__global__ void neuronSum(int layerSize, int numWeights, double* layerBiases, double* weightProducts, double* output) {
    int index = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    if (index < layerSize) {
        double sum = 0;
        for (int i = 0; i < numWeights; i++) {
            sum += weightProducts[index * numWeights + i];
        }
        output[index] = sum + layerBiases[index];
    }
}

// multiplies each weight to the corresponding input
__global__ void layerWeightedSumCUDA(int layerSize, int numWeights, double* layerWeights, double* input, double* output) {
    int index = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    if (index < numWeights * layerSize) {
        output[index] = input[index % numWeights] * layerWeights[index];
    }
}

// softmax activation function
__global__ void softmaxCUDA(int numOutputs, double denominator, double* neuronOutputs, double* softmaxResult) {
    int index = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    if (index < numOutputs) {
        softmaxResult[index] = exp(neuronOutputs[index]) / denominator;
    }
}

// softmax derivative function
__global__ void softmaxDerivCUDA(int layerSize, double* activatedOutputs, double* targets, double* derivs) {
    int index = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    if (index < layerSize) {
        derivs[index] = (activatedOutputs[index] - targets[index]);
    }
}

// ReLu activation function
__global__ void reluCUDA(double* neuronOutputs, double* activatedOutputs, int layerSize) {
    int index = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    if (index < layerSize) {
        if (neuronOutputs[index] > 0) {
            activatedOutputs[index] = neuronOutputs[index];
        }
        else {
            activatedOutputs[index] = 0;
        }
    }
}

//ReLu derivation, defined as 0 at x=0
__global__ void reluDerivCUDA(int layerSize, double* neuronOutputs, double* derivs) {
    int index = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    if (index < layerSize) {
        if (neuronOutputs[index] > 0) {
            derivs[index] *= 1;
        }
        else {
            derivs[index] *= 0;
        }
    }
}

// computes the gradient for all neurons in a layer
__global__ void hiddenGradient(int curLayerSize, int nextLayerSize, int numNextWeights, double* nextLayerWeights, double* nextDeltas, double* derivs) {
    int index = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    double total = 0;
    if (index < curLayerSize) {
        for (int curN = 0; curN < nextLayerSize; curN++) {
            total += nextLayerWeights[curN * numNextWeights + index] * nextDeltas[curN];
        }
    }
    derivs[index] = total;
}

// updates all weights for all neurons in a layer
__global__ void updateLayerWeights(int layerSize, int numWeights, double learningRate, double momentum, double* layerWeights, double* layerDeltas, double* weightGradients, double* layerInputs) {
    int index = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    int neuronIndex = floor(index / (double)numWeights);
    if (index < layerSize * numWeights) {
        double result = (layerInputs[index % numWeights] * layerDeltas[neuronIndex] * learningRate);
        layerWeights[index] -= result;
        weightGradients[index] = result;
    }
}

// updates all biases in a layer
__global__ void updateLayerBiases(int layerSize, double learningRate, double momentum, double* layerBiases, double* layerDeltas, double* biasGradients) {
    int index = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    if (index < layerSize) {
        //double gradient = (layerDeltas[index] * learningRate) + (biasGradients[index] * momentum);
        double gradient = (layerDeltas[index] * learningRate);
        layerBiases[index] -= gradient;
        biasGradients[index] = gradient;
    }
}

// min-max normalization
__global__ void dataNormalize(int numInputs, int numColumns, double* colMinimums, double* colMaximums, double* data) {
    int curColumn = (int) (threadIdx.x + blockIdx.x * blockDim.x);
    if (curColumn < numColumns) {
        for (int i = 0; i < numInputs; i++) {
            if (colMaximums[curColumn] - colMinimums[curColumn] == 0) {
                data[curColumn + i * numColumns] = 0;
            }
            else {
                data[curColumn + i * numColumns] = (data[curColumn + i * numColumns] - colMinimums[curColumn]) / (colMaximums[curColumn] - colMinimums[curColumn]);
            }
        }
    }

}

