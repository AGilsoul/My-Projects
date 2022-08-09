//
// Created by agils on 8/7/2022.
//
#include "../include/LinearRegressor.h"
#include "../include/TicTacToeEngine.h"
#include "../src/LinearRegressor.cpp"


#pragma once

class ReinforcementModel {
public:
    ReinforcementModel(int gameGrid, double learningRate=0.001, double discountFactor=0.9);

    void trainModel(int numIterations);

    void makeMove(TicTacToeEngine& game, int player);

    double reward(int* board, int player);

private:
    int arrSum(int* arr, int n, int step=1);

    LinearRegressor<int> model1;
    LinearRegressor<int> model2;
    std::uniform_real_distribution<double> qUnif;
    std::uniform_int_distribution<int> moveUnif;
    double lr;
    double df;
    int gameGrid;
    int spaceCount;

};

