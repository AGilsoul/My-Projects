//
// Created by agils on 8/7/2022.
//

#include "../include/TicTacToeEngine.h"

TicTacToeEngine::TicTacToeEngine() {
    for (int i = 0; i < NUMSPACES; i++) {
        movesRemaining[i] = i;
    }
}

bool TicTacToeEngine::move(int row, int col, int player) {
    if (board[row * DIM + col] == 0) {
        board[row * DIM + col] = player;
        movesRemaining.erase(remove(movesRemaining.begin(), movesRemaining.end(), row * DIM + col), movesRemaining.end());
        return true;
    }
    return false;
}

bool TicTacToeEngine::moveLinear(int pos, int player) {
    if (board[pos] == 0) {
        board[pos] = player;
        movesRemaining.erase(remove(movesRemaining.begin(), movesRemaining.end(), pos), movesRemaining.end());
        return true;
    }
    return false;
}

int* TicTacToeEngine::getState() {
    return board;
}

int TicTacToeEngine::checkState() {
    // horizontal check
    for (int row = 0; row < DIM; row++) {
        if (abs(arrSum(board + row*DIM, DIM)) == 3) {
            return board[row * DIM];
        }
    }
    // vertical check
    for (int col = 0; col < DIM; col++) {
        if (abs(arrSum(board + col, DIM, DIM)) == 3) {
            return board[col];
        }
    }
    // diagonal check
    if (abs(arrSum(board, DIM, DIM+1)) == DIM) {
        return board[0];
    }
    if (abs(arrSum(board + DIM - 1, DIM, DIM-1) ) == DIM) {
        return board[DIM-1];
    }
    return 0;
}

bool TicTacToeEngine::gameOver() {
    return movesRemaining.empty() || checkState() != 0;
}

void TicTacToeEngine::showBoard() {
    for (int row = 0; row < DIM; row++) {
        for (int col = 0; col < DIM; col++) {
            cout << setw(3) << board[row * DIM + col];
        }
        cout << endl;
    }
}

void TicTacToeEngine::reset() {
    for (int i = 0; i < NUMSPACES; i++) {
        board[i] = 0;
    }
    for (int i = 0; i < NUMSPACES; i++) {
        movesRemaining[i] = i;
    }
}

TicTacToeEngine::~TicTacToeEngine() {}

int TicTacToeEngine::arrSum(int* arr, int n, int step) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i * step];
    }
    return sum;
}
