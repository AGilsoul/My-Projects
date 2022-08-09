//
// Created by agils on 8/7/2022.
//

#include <cstdlib>
#include <random>
#include <algorithm>
#include <chrono>
#include <windows.h>
#include <iostream>
#include <iomanip>

#pragma once

using std::time;
using std::cout;
using std::endl;

template <typename T>
class LinearRegressor {
public:
    LinearRegressor<T>(int numInputs, double learningRate);

    double predict(T* dataIn);

    void fit(T* trainData, double* trainTargets, int dataCount, int iterations=1);

    void gradientDescent(T* data, double target);

    void applyGradients();

    double eval(T* evalData, double* evalTargets, int dataCount);

    double sigmoid(double input);

    void initWeights();

    std::default_random_engine rng;
    int numWeights;
    double lr;
    double* weights;
    double* deltaW;
    double bias;
    double deltaB;
    double lastSum;
    double activatedSum;
};

