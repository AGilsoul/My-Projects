//
// Created by agils on 8/5/2022.
//
#include <iostream>
#include <iomanip>

#pragma once

#define BOARDDIM 19

using std::ostream;
using std::cout;
using std::endl;
using std::setw;

class Board {
public:
    Board() {}
    
    bool move(int row, int col, int color);

    friend ostream& operator<<(ostream& out, const Board& b);

private:
    int requestState(int row, int col);
    // 0-Empty; 1-white; 2-black
    int boardStates[361] = { 0 };
    
    
};
