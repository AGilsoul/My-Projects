//
// Created by agils on 8/7/2022.
//


#include "../include/LinearRegressor.h"

template <typename T>
LinearRegressor<T>::LinearRegressor(int numInputs, double learningRate) {
    numWeights = numInputs;
    this->lr = learningRate;
    weights = new double[numWeights];
    deltaW = new double[numWeights];
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
    initWeights();
}

template <typename T>
double LinearRegressor<T>::predict(T* dataIn) {
    /*
    cout << "data: ";
    for (int i = 0; i < numWeights; i++) {
        cout << std::setw(4) << dataIn[i] << " ";
    }
    cout << endl;
    cout << "weights: ";
    for (int i = 0; i < numWeights; i++) {
        cout << std::setw(4) << weights[i] << " ";
    }
    cout << "bias: " << bias;
    cout << endl;
     */
    double sum = bias;
    for (int i = 0; i < numWeights; i++) {
        sum += weights[i] * dataIn[i];
    }
    lastSum = sum;
    //activatedSum = sigmoid(sum);
    activatedSum = sum;
    //cout << "sum: " << activatedSum << endl;
    return sum;
    //return activatedSum;
}

template <typename T>
void LinearRegressor<T>::fit(T* trainData, double* trainTargets, int dataCount, int iterations) {
    for (int i = 0; i < iterations; i++) {
        for (int d = 0; d < dataCount; d++) {
            gradientDescent(trainData + d * numWeights, trainTargets[d]);
        }
    }
}

template <typename T>
void LinearRegressor<T>::gradientDescent(T* data, double target) {
    double modelOut = predict(data);
    //double delta = sigmoid(lastSum) * (1-sigmoid(lastSum)) * 2 * (modelOut - target);
    double delta = 2 * (modelOut - target);
    for (int w = 0; w < numWeights; w++) {
        deltaW[w] = delta * data[w];
    }
    deltaB = delta;
    applyGradients();
}

template <typename T>
void LinearRegressor<T>::applyGradients() {
    for (int w = 0; w < numWeights; w++) {
        weights[w] -= deltaW[w] * lr;
    }
    bias -= deltaB * lr;
}


template <typename T>
double LinearRegressor<T>::eval(T* evalData, double* evalTargets, int dataCount) {
    double mse = 0;
    for (int d = 0; d < dataCount; d++) {
        double modelOut = predict(evalData + d * numWeights);
        mse += pow(modelOut - evalTargets[d], 2);
    }
    return mse / dataCount;
}

template <typename T>
double LinearRegressor<T>::sigmoid(double input) {
    return 1/(1+exp(-input));
}

template <typename T>
void LinearRegressor<T>::initWeights() {
    // normalized xavier weight initialization
    std::uniform_real_distribution<double> unif(-1 * sqrt(6.0) / sqrt(numWeights + 1), sqrt(6.0) / sqrt(numWeights + 1));
    for (int i = 0; i < numWeights; i++) {
        weights[i] = unif(rng);
    }
    bias = unif(rng);
}



