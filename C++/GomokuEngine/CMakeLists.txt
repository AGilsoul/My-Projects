cmake_minimum_required(VERSION 3.21)
project(GomokuEngine)

set(CMAKE_CXX_STANDARD 14)

add_executable(GomokuEngine main.cpp Board.cpp Board.h Game.cpp Game.h)
