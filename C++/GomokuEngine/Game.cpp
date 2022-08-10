//
// Created by agils on 8/6/2022.
//

#include "Game.h"
#include <iostream>

using std::cout;
using std::endl;

Game::Game() {
    gameBoard = Board();
    gameState = true;
    playerTurn = 1;
}

bool Game::makeMove(int row, int col, int color) {
    cout << "Cur player turn: " << playerTurn << endl;
    if (gameBoard.move(row, col, color) && playerTurn == color) {
        playerTurn = playerTurn % 2 + 1;
        return true;
    }
    return false;
}

void Game::showBoard() {
    cout << gameBoard << endl;
}
