//
// Created by agils on 8/6/2022.
//
#include "Board.h"

#pragma once

class Game {
public:
    Game();

    bool makeMove(int row, int col, int color);

    void showBoard();

private:
    Board gameBoard;
    bool gameState;
    int playerTurn;

};
