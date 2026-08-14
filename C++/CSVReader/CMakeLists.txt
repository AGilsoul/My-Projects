cmake_minimum_required(VERSION 3.21)
project(CSVReader)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)


#add_executable(CSVReader
#        include/csvReader.h
#        include/hashtable.h
#        src/csvReader.cpp
#        src/main.cpp)

add_library(CSVReader include/csvReader.h include/hashtable.h src/csvReader.cpp)
