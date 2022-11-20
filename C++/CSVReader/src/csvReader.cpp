//
// Created by agils on 11/19/2022.
//

#include "../include/csvReader.h"


csvReader::csvReader(const string& fileName): fileName(fileName) {
    fstream fin(fileName, fstream::in);
    if (!fin) {
        throw runtime_error("File " + fileName + " not found");
    }
    readHeaders(fin);
    headerHash = hashtable(numColumns);
    for (int i = 0; i < numColumns; i++) {
        headerHash.insert(columnHeaders[i], i);
    }
    readData(fin);
}

vector<double> csvReader::toDouble(vector<string> col) {
    vector<double> doubleData(col.size());
    for (int i = 0; i < col.size(); i++) {
        doubleData[i] = stod(col[i]);
    }
    return doubleData;
}

vector<string>& csvReader::operator[](const int index) {
    return data[index];
}

vector<string>& csvReader::operator[](const string& header) {
    int headerIndex = headerHash[header];
    if (headerIndex == -1) {
        throw runtime_error("No column with header \"" + header + "\"");
    }
    return dataTranspose[headerIndex];
}


void csvReader::readHeaders(fstream& fin) {
    char ch = 0;
    string buffer;
    // read column headers
    while (ch != '\n') {
        fin >> noskipws >> ch;
        if (ch != ',' && ch != '\n') {
            buffer += ch;
        }
        else {
            numColumns++;
            columnHeaders.push_back(buffer);
            buffer = "";
        }
    }
    columnHeaders.push_back(buffer);
}

void csvReader::readData(fstream& fin) {
    char ch = 0;
    string buffer;

    while (!fin.eof()) {
        vector<string> curRow(numColumns);
        buffer = "";
        int valCount = 0;
        while (true) {
            fin >> noskipws >> ch;
            if (ch == ',' || ch == '\n' || fin.eof()) {
                curRow[valCount] = buffer;
                buffer = "";
                valCount++;
                if (ch == '\n' || fin.eof()) {
                    break;
                }
            }
            else {
                buffer += ch;
            }
        }
        data.push_back(curRow);
    }
    data.shrink_to_fit();
    dataTranspose = transpose<string>(data);
}


ostream& operator<<(ostream& out, const csvReader& reader) {
    out << "[ ";
    for (int h = 0; h < reader.numColumns; h++) {
        out << reader.columnHeaders[h];
        if (h != reader.numColumns - 1) {
            out << " , ";
        }
    }
    out << " ]\n";

    for (int r = 0; r < reader.data.size(); r++) {
        out << "[ ";
        for (int c = 0; c < reader.numColumns; c++) {
            out << reader.data[r][c];
            if (c != reader.numColumns - 1) {
                out << " , ";
            }
        }
        out << " ]\n";
    }
    return out;
}



