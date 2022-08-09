#include <iostream>
#include "../include/TicTacToeEngine.h"
#include "../include/LinearRegressor.h"
#include "../include/ReinforcementModel.h"

int main() {

    /*
    LinearRegressor<int> model(1, 0.01);
    int inputs[] = {1, 2, 3, 4, 5, 6, 7};
    double targets[] = {1, 2, 3, 4, 5, 6, 7};
    model.fit(inputs, targets, 7, 10);
    cout << "MSE: ";
    cout << model.eval(inputs, targets, 7) << endl;
    */

    ReinforcementModel model(3);
    cout << "Training..." << endl;
    model.trainModel(100000);

    TicTacToeEngine game;
    int playerMove;
    //game.showBoard();
    while (!game.gameOver()) {
        model.makeMove(game, P2);
        game.showBoard();
        cout << "make a move: ";
        std::cin >> playerMove;
        cout  << endl;
        game.moveLinear(playerMove, P1);

    }

    return 0;
}

// self playing algorithm

/*
 *  selfPlay(player, initialState, model) {
 *      e(0) = initialGradients
 *      S(0) = initialState
 *
 *      t = 0
 *
 *      while (game not over) {
 *          V(old) = modelPrediction(S(t))
 *          q = random(0,1)
 *          V(S(t+1)) = { 0 if game is over, modelPrediction(S(t+1)) otherwise
 *          if (q < 0.25) {
 *              S(t+1) = randomMove
 *          }
 *          else {
 *              curState =
 *              nextMove that maximizes p*(reward(S(t+1) + γ*V(S(t+1)))
 *          }
 *          E(t) = { 0 if q < 0.25, r(S(t+1)) + γ*V(S(t+1)) - V(old) otherwise
 *
 *          adjust weights (add E(t) * e(t))
 *          e(t+1) = γ*λ*e(t) + modelPrediction(S(t+1))
 *          p *= -1
 *          t++
 *      }
 *  }
 *
 *
 */

