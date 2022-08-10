//
// Created by agils on 8/5/2022.
//

#include "Board.h"

bool Board::move(int row, int col, int color) {
    if (this->requestState(row, col) == 0) {
        this->boardStates[row * BOARDDIM + col] = color;
        return true;
    }
    return false;
}

ostream& operator<<(ostream& out, const Board& b) {
    out << " ";
    for (int i = 0; i < BOARDDIM; i++) {
        out << " " << setw(2) << i;
    }
    out << endl;
    for (int x = 0; x < BOARDDIM; x++) {
        if (x != 0) {
            out << "  ";
            for (int i = 0; i < BOARDDIM; i++) {
                out << "  |";
            }
            out << endl;
        }
        out << setw(2) << x;
        for (int y = 0; y < BOARDDIM; y++) {
            out << "--";
            switch (b.boardStates[x * BOARDDIM + y]) {
                case 1:
                    out << "W";
                    break;
                case 2:
                    out << "B";
                    break;
                default:
                    out << "0";
                    break;
            }
        }
        out << "--" << endl;
    }
    return out;
}

int Board::requestState(int row, int col) {
    return boardStates[row * BOARDDIM + col];
}
