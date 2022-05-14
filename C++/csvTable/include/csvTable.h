//
// Created by agils on 5/12/2022.
//

#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

using std::string;
using std::cout;
using std::vector;
using std::endl;

// row struct
template <typename T>
struct row {
    row(vector<T> data) {
        this->rowData = data;
    }

    T operator[](int index) {
        return rowData[index];
    }

    void removeColumn(int index) {
        std::remove(rowData.begin(), rowData.end(), index);
    }

    void addColumn() {
        rowData.reserve(rowData.size() + 1);
    }

    void addColumn(T newVal) {
        rowData.push_back(newVal);
    }

    void toString() {
        cout << "[";
        for (int i = 0; i < rowData.size(); i++) {
            cout << std::setw(15) << std::right << rowData[i];
            if (i != rowData.size() - 1) {
                cout << ", ";
            }
        }
        cout << "]\n";
    }

    double mean() {
        try {
            double total = 0;
            for (int i = 0; i < rowData.size(); i++) {
                total += rowData[i];
            }
            return total / rowData.size();
        }
        catch (std::exception e) {
            cout << e.what() << endl;
        }
    }

    double stdDev() {
        try {
            double total = 0;
            double avg = mean();
            for (int i = 0; i < rowData.size(); i++) {
                total += pow((rowData[i] - avg), 2);
            }
            return sqrt(total / rowData.size());
        }
        catch (std::exception e) {
            cout << e.what() << endl;
        }

    }

    friend std::ostream& operator<<(std::ostream& out, row& r) {
        out << "[";
        for (int i = 0; i < r.rowData.size(); i++) {
            out << std::setw(15) << std::right << r.rowData[i];
            if (i != r.rowData.size() - 1) {
                out << ", ";
            }
        }
        out << "]\n";
        return out;
    }

    vector<T> rowData;
};

// csvTable class
class csvTable {
public:
    // constructor for csvTable
    csvTable(string fileName, string tableName="CSV Table"): tableName(tableName) {
        std::ifstream fin(fileName);
        genColumnLabels(fin);
        readFile(fin);
        fin.close();
    }

    // creates a row struct from a column of data
    row<string> getColumn(int index) {
        vector<string> colData;
        for (int i = 0; i < numRows; i++) {
            colData.push_back(rows[i][index]);
        }
        row<string> curColumn(colData);
        return curColumn;
    }



    row<double> rowToDouble(row<string> r) {
        vector<double> vec;
        for (int i = 0; i < r.rowData.size(); i++) {
            vec.push_back(stod(r[i]));
        }
        row<double> tempRow(vec);
        return tempRow;
    }

    // prints out all column labels
    void printColumns() {
        columnLabelsString();
    }

    // prints out entire table
    void printTable() {
        cout << tableName << endl;
        columnLabelsString();
        for (int i = 0; i < numRows; i++) {
            rows[i].toString();
        }
    }

    void printColumnInfo(string columnName) {
        int colIndex = -1;
        for (int i = 0; i < numColumns; i++) {
            if (columnLabels[i] == columnName) {
                colIndex = i;
                break;
            }
        }
        if (colIndex == -1) {
            cout << "Column not found" << endl;
        }
        else {
            printColumnInfo(colIndex);
        }
    }

    void printColumnInfo(int index) {
        auto curColumn = rowToDouble(getColumn(index));
        cout << "Column Name: " << columnLabels[index] << endl;
        cout << "Mean: " << curColumn.mean() << endl;
        cout << "Standard Deviation: " << curColumn.stdDev() << endl;
    }

    // overloaded [] operator for index-based access
    row<string> operator[](int index) {
        return rows[index];
    }

private:
    // prints out all column labels
    void columnLabelsString() {
        cout << "[";
        for (int i = 0; i < numColumns; i++) {
            cout << std::setw(15) << std::right << columnLabels[i];
            if (i != numColumns - 1) {
                cout << ", ";
            }
        }
        cout << "]\n";
    }

    // generates column labels
    void genColumnLabels(std::ifstream& fin) {
        string curString;
        int numColumns = 0;
        vector<string> columns;
        bool looking = true;
        while (!fin.eof() && looking) {
            fin >> curString;
            if (curString.at(curString.length() - 1) != ',') {
                numColumns++;
                columns.push_back(curString);
                getline(fin, curString);
                looking = false;
            }
            else {
                numColumns++;
                curString.pop_back();
                columns.push_back(curString);
            }
        }
        this->numColumns = numColumns;
        this->columnLabels = columns;
    }

    // reads the entirety of a file after the column headers
    void readFile(std::ifstream& fin) {
        string curString;
        while (!fin.eof()) {
            vector<string> data;
            data.reserve(numColumns);
            for (int i = 0; i < numColumns - 1; i++) {
                std::getline(fin, curString, ',');
                data.push_back(curString);
            }
            std::getline(fin, curString, '\n');
            data.push_back(curString);
            row<string> curRow(data);
            rows.push_back(curRow);
            numRows++;
        }
    }

    int numColumns = 0;
    int numRows = 0;
    string tableName;
    vector<string> columnLabels;
    vector<row<string>> rows;
};

