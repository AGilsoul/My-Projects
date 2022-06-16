#include <iostream>
#include <ostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include "../include/ConvNet.h"

using std::cin;
using std::ifstream;
using std::ios;

void readMnistFile(vector<tensor>& testData, vector<int>& expected, int numChannels=1, int numDuplicationChannels=1);

int main() {
    string placeHolder;
    int actualDataChannels = 1;
    int numDupChannels = 1;
    cout << "Constructing Convolutional Neural Network..." << endl;
    ConvNet cnn(numDupChannels, 28, 0.001, 0.9, true);
    cnn.addKernel(3, 5, true, true);
    cnn.addKernel(9, 5, true, true);
    cnn.addHiddenLayer(128);
    cnn.addOutputLayer(10);
    cout << "Done" << endl << endl;

    vector<tensor> data;
    vector<int> labels;
    cout << "Reading file..." << endl;
    readMnistFile(data, labels, actualDataChannels, numDupChannels);
    cout << "Normalizing..." << endl;
    normalize(data, {{0,255}});
    cout << "Splitting Data..." << endl << endl;
    vector<tensor> trainData(data.begin(), data.begin() + 40000);
    vector<int> trainLabels(labels.begin(), labels.begin() + 40000);
    vector<tensor> testData(data.begin() + 40000, data.end());
    vector<int> testLabels(labels.begin() + 40000, labels.end());
    cout << "Fitting Model..." << endl;
    cnn.fitModel(trainData, trainLabels, 5, testData, testLabels);
    cout << "Press x to continue" << endl;
    cin >> placeHolder;
    return 0;
}


void readMnistFile(vector<tensor>& testData, vector<int>& expected, int numChannels, int numDuplicationChannels) {
    //784 data columns, one label column at beginning
    //strings to be used for reference and assignment of values when reading the file and assigning to the string list sList
    string sList[785];
    for (unsigned int i = 0; i < 785; i++) {
        string temp = "";
        sList[i] = temp;
    }
    //Reads from the file "mnist_train.csv"
    ifstream fin("../res/mnist_train.csv", ios::in);
    //vector<string> labels;
    int listSize = sizeof(sList) / sizeof(sList[0]);
    while (!fin.eof()) {
        vector<double> dData;
        int result;
        for (unsigned int i = 0; i < listSize; i++) {
            if (i != listSize - 1) {
                getline(fin, sList[i], ',');
            }
            else {
                getline(fin, sList[i], '\n');
            }

            if (i != 0) {
                dData.push_back(stod(sList[i]));
            }
            else {
                result = stoi(sList[i]);
            }
        }
        vector<double> finData;
        for (int chan = 0; chan < numDuplicationChannels; chan++) {
            finData.insert(finData.end(), dData.begin(), dData.end());
        }
        expected.push_back({result});
        testData.push_back(reshapeVector(finData, numChannels * numDuplicationChannels, 28, 28));
    }
    fin.close();
}
