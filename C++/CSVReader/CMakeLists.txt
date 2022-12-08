cmake_minimum_required(VERSION 3.21)
project(CSVReader)

set(CMAKE_CXX_STANDARD 14)

add_library(CSVReader STATIC src/csvReader.cpp)
include_directories(include)
