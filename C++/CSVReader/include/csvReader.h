//
// Created by agils on 11/19/2022.
//

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <memory>
#include "hashtable.h"

using std::string;
using std::vector;
using std::fstream;
using std::noskipws;
using std::cout;
using std::endl;
using std::getline;
using std::ostream;
using std::stod;
using std::exception;
using std::runtime_error;

class csvReader {
public:

    csvReader(const string& fileName);

    vector<double> toDouble(vector<string> col);

    vector<string>& operator[](const int index);

    vector<string>& operator[](const string& header);

    friend ostream& operator<<(ostream& out, const csvReader& reader);


private:
    void readHeaders(fstream& fin);

    void readData(fstream& fin);

    template<typename T>
    vector<vector<T>> transpose(vector<vector<T>> mat) {
        vector<vector<T>> newMat(mat[0].size());
        for (int col = 0; col < mat[0].size(); col++) {
            vector<T> newRow(mat.size());
            for (int row = 0; row < mat.size(); row++) {
                newRow[row] = mat[row][col];
            }
            newMat[col] = newRow;
        }
        return newMat;
    }

    hashtable headerHash;
    vector<string> columnHeaders;
    vector<vector<string>> data;
    vector<vector<string>> dataTranspose;
    string fileName;
    int numColumns = 0;


};

