cmake_minimum_required(VERSION 3.21)
project(GameAI)

set(CMAKE_CXX_STANDARD 14)

add_executable(GameAI src/main.cpp src/TicTacToeEngine.cpp include/TicTacToeEngine.h src/LinearRegressor.cpp include/LinearRegressor.h src/ReinforcementModel.cpp include/ReinforcementModel.h)
