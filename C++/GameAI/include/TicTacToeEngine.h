//
// Created by agils on 8/7/2022.
//
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#pragma once

#define P1 1
#define P2 -1
#define DIM 3
#define NUMSPACES 9

using std::cout;
using std::endl;
using std::setw;
using std::vector;
using std::remove;

class TicTacToeEngine {
public:
    TicTacToeEngine();

    bool move(int row, int col, int player);

    bool moveLinear(int pos, int player);

    int* getState();

    vector<int> getMoves() { return movesRemaining; }

    int checkState();

    bool gameOver();

    void showBoard();

    void reset();

    ~TicTacToeEngine();


    int arrSum(int* arr, int n, int step=1);

    int board[NUMSPACES] = { 0 };
    vector<int> movesRemaining = vector<int>(NUMSPACES);
};

