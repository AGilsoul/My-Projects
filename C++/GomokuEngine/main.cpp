#include <iostream>
#include "Game.h"

using std::cout;
using std::endl;

enum Color{none=0, white, black};

int main() {
    Game testGame;
    Color p1 = white;
    Color p2 = black;
    testGame.makeMove(10, 10, p1);
    testGame.makeMove(4, 18, p2);
    testGame.showBoard();

    return 0;
}
