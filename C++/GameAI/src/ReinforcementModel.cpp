//
// Created by agils on 8/7/2022.
//

#include "../include/ReinforcementModel.h"

ReinforcementModel::ReinforcementModel(int gridDim, double learningRate, double discountFactor): model1(pow(gridDim, 2), learningRate), model2(pow(gridDim, 2), learningRate) {
    lr = learningRate;
    df = discountFactor;
    gameGrid = gridDim;
    spaceCount = pow(gridDim, 2);
    qUnif = std::uniform_real_distribution<double>(0, 1);
}

void ReinforcementModel::trainModel(int numIterations) {
    int p1Wins = 0;
    int p2Wins = 0;
    int draws = 0;
    for (int i = 0; i < numIterations; i++) {
        TicTacToeEngine game;
        int p = P1;
        int St[spaceCount];
        int gameStates[9][9];
        int numStates = 0;
        //double vOld;
        //double vNew;
        LinearRegressor<int>* curModel = &model1;

        //cout << "starting game" << endl;
        int moveCount = 0;
        while (!game.gameOver()) {
            if (p == 1) {
                curModel = &model1;
            }
            else {
                curModel = &model2;
            }
            //cout << "copying moves to state" << endl;
            std::copy(game.getState(), game.getState() + spaceCount, St);
            //cout << "getting moves" << endl;
            vector<int> remainingMoves = game.getMoves();
            //cout << "making unif" << endl;
            moveUnif = std::uniform_int_distribution<int>(0, remainingMoves.size()-1);
            //cout << "getting rand number" << endl;
            double q = qUnif(curModel->rng);
            if (q < 0.4) {
                //cout << "rand move" << endl;
                int nextMove = remainingMoves[moveUnif(curModel->rng)];
                game.moveLinear(nextMove, p);
            }
            else {
                //cout << "non rand move" << endl;
                int maxMove = -1;
                double maxReward = 0;
                for (int m = 0; m < remainingMoves.size(); m++) {
                    St[remainingMoves[m]] = p;
                    double curReward = pow(df, moveCount) * curModel->predict(St);
                    St[remainingMoves[m]] = 0;
                    if (curReward > maxReward || maxMove == -1) {
                        maxMove = m;
                        maxReward = curReward;
                    }
                }
                //cout << "making move" << endl;
                game.moveLinear(remainingMoves[maxMove], p);
                //cout << "copying move state" << endl;
                //cout << numStates << endl;
                std::copy(game.getState(), game.getState()+spaceCount, gameStates[numStates]);
                //cout << setw(3) << p << " chose " << remainingMoves[maxMove] << " - reward: " << reward(St, p) << " - state reward: " << maxReward << endl;
                //cout <<
                numStates++;
            }
            moveCount++;
            p *= -1;
        }
        //cout << "updating" << endl;
        int winner = game.checkState();
        //cout << "game end: " << winner << endl;
        // update models
        double r1;
        double r2;
        if (winner == 1) {
            r1 = 1;
            r2 = 0;
        }
        else if (winner == -1) {
            r1 = 0;
            r2 = 1;
        }
        else {
            r1 = 0.1;
            r2 = 0.5;
        }
        for (int s = numStates-1; s >= 0; s--) {
            int curState = numStates - s - 1;
            model1.gradientDescent(gameStates[curState], r1 * pow(df, s));
            model2.gradientDescent(gameStates[curState], r2 * pow(df, s));
        }

        //cout << "done updating" << endl;
        if (winner == P1) {
            p1Wins++;
        }
        else if (winner == P2) {
            p2Wins++;
        }
        else {
            draws++;
        }
    }
    cout << "p1: " << p1Wins << " p2: " << p2Wins << " draws: " << draws << endl << endl;
}

void ReinforcementModel::makeMove(TicTacToeEngine& game, int player) {
    int St[spaceCount];
    std::copy(game.getState(), game.getState() + spaceCount, St);
    auto remainingMoves = game.getMoves();
    int maxMove = -1;
    double maxReward = 0;
    LinearRegressor<int> *curModel = &model1;
    if (player == P2) {
        curModel = &model2;
    }
    for (int m = 0; m < remainingMoves.size(); m++) {
        St[remainingMoves[m]] = player;
        double curReward = curModel->predict(St);
        St[remainingMoves[m]] = 0;
        if (curReward > maxReward || maxMove == -1) {
            maxMove = remainingMoves[m];
            maxReward = curReward;
        }
    }
    cout << "max move: " << maxMove << endl;
    game.moveLinear(maxMove, player);
}

double ReinforcementModel::reward(int* board, int curPlayer) {
    double result;
    // horizontal check
    for (int row = 0; row < gameGrid; row++) {
        if (abs(arrSum(board + row*gameGrid, gameGrid)) == 3) {
            result = board[row * gameGrid];
        }
    }
    // vertical check
    for (int col = 0; col < gameGrid; col++) {
        if (abs(arrSum(board + col, gameGrid, gameGrid)) == 3) {
            result = board[col];
        }
    }
    // diagonal check
    if (abs(arrSum(board, gameGrid, gameGrid+1)) == gameGrid) {
        result = board[0];
    }
    if (abs(arrSum(board + gameGrid - 1, gameGrid, gameGrid-1) ) == gameGrid) {
        result = board[gameGrid-1];
    }
    if (result == curPlayer) {
        return 1;
    }
    else if (result == -curPlayer) {
        return 0;
    }
    return 0;
}

// self playing algorithm
/*
 *  selfPlay(player, initialState, model) {
 *      e(0) = initialGradients
 *      S(0) = initialState
 *      t = 0
 *      while (game not over) {
 *          V(old) = modelPrediction(S(t))
 *          q = random(0,1)
 *          if (q < 0.25) {
 *              S(t+1) = randomMove
 *          }
 *          else {
 *              curState =
 *              nextMove that maximizes p*(reward(S(t+1)+ γ*V(S(t+1)))
 *          }
 *          V(S(t+1)) = { 0 if game is over, modelPrediction(S(t+1)) otherwise
 *          E(t) = { 0 if q < 0.25, r*S(t+1) + γ*V(S(t+1)) - V(old) otherwise
 *
 *          adjust weights (add E(t) * e(t))
 *          e(t+1) = γ*λ*e(t) + modelPrediction(S(t+1))
 *          p *= -1
 *          t++
 *      }
 *  }
 */

int ReinforcementModel::arrSum(int* arr, int n, int step) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i * step];
    }
    return sum;
}
