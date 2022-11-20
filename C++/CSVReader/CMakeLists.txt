cmake_minimum_required(VERSION 3.21)
project(CSVReader)

set(CMAKE_CXX_STANDARD 14)

add_executable(CSVReader src/main.cpp src/csvReader.cpp include/csvReader.h include/hashtable.h)
