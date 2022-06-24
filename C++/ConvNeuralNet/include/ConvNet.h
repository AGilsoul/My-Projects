//
// Created by agils on 6/10/2022.
//

#pragma once

#include "../include/alg_stopwatch.h"
#include "../include/ConvLayer.h"
#include <thread>
#include <windows.h>

using std::make_shared;
using std::thread;
using std::setprecision;
using std::shared_ptr;
using std::ostream;

class ConvNet {
public:
    ConvNet(int inputChannelCount, int inputDim, double lr, double m=0.0): inputChannelCount(inputChannelCount), inputDim(inputDim), lr(lr), m(m) {}

    void addKernel(int numKernels, int kernelDim, bool padding=true, bool pooling=false, const string& poolType="avg") {
        if (kernels.empty()) {
            numConnections += numKernels * pow(kernelDim, 2) * inputChannelCount;
            kernels.push_back(make_shared<KernelLayer>(numKernels, kernelDim, inputChannelCount, inputDim, lr, m, padding, pooling, poolType));
        }
        else {
            numConnections += numKernels * pow(kernelDim, 2) * kernels[kernels.size() - 1]->getNumKernels();
            kernels.push_back(make_shared<KernelLayer>(numKernels, kernelDim, kernels[kernels.size() - 1]->getNumKernels(), kernels[kernels.size() - 1]->getOutputDim(), lr, m, padding, pooling, poolType));
        }

    }

    void addHiddenLayer(int numNeurons) {
        if (kernels.empty()) {
            throw exception("Add kernels before dense layers");
        }

        if (hiddenLayers.empty()) {
            int numInputs = kernels[kernels.size() - 1]->getNumKernels() * pow(kernels[kernels.size() - 1]->getOutputDim(), 2);
            hiddenLayers.push_back(make_shared<HiddenLayer>(numNeurons, numInputs, lr, m));
            numConnections += numNeurons * numInputs;
        }
        else {
            int numInputs = hiddenLayers[hiddenLayers.size() - 1]->getNumNeurons();
            hiddenLayers.push_back(make_shared<HiddenLayer>(numNeurons, numInputs, lr, m));
            numConnections += numNeurons * numInputs;
        }
    }

    void addOutputLayer(int numNeurons) {
        if (kernels.empty()) {
            throw exception("Add kernels before dense layers");
        }

        if (hiddenLayers.empty()) {
            int numInputs = kernels[kernels.size() - 1]->getNumKernels() * pow(kernels[kernels.size() - 1]->getOutputDim(), 2);
            outLayer = make_shared<OutputLayer>(numNeurons, numInputs, lr, m);
            numConnections += numNeurons * numInputs;
        }
        else {
            int numInputs = hiddenLayers[hiddenLayers.size() - 1]->getNumNeurons();
            outLayer = make_shared<OutputLayer>(numNeurons, numInputs, lr, m);
            numConnections += numNeurons * numInputs;
        }
    }

    int predict(tensor data) {
        auto res = propagate(data);
        int maxIndex = -1;
        double maxVal = 0;
        for (int i = 0; i < res.size(); i++) {
            if (res[i] > maxVal) {
                maxVal = res[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    vector<double> propagate(tensor dataIn) {
        for (auto & kernel : kernels) {
            dataIn = kernel->propagate(dataIn);
        }
        vector<double> flatData = flattenTensor(dataIn);
        for (auto & hiddenLayer : hiddenLayers) {
            flatData = hiddenLayer->propagate(flatData);
        }
        return outLayer->propagate(flatData);
    }

    void backPropagate(vector<tensor> trainData, vector<int> trainLabels, int iterations) {
        *this->progressGoal = iterations * trainData.size();
        // set pre kernel delta size
        vector<double> preKernDeltas(kernels[kernels.size() - 1]->getNumKernels() * pow(kernels[kernels.size() - 1]->getOutputDim(), 2));
        // one-hot encode all labels
        auto labels = oneHotEncode(std::move(trainLabels));
        // for every iteration
        for (int iters = 0; iters < iterations; iters++) {
            // for every data point
            for (int dIndex = 0; dIndex < trainData.size(); dIndex++) {
                *this->curProgress += 1;
                vector<double> curLabel = labels[dIndex];
                // calculate network output
                vector<double> result = propagate(trainData[dIndex]);
                // calculate output deltas
                vector<double> flatDeltas = outLayer->backPropagate(curLabel);

                // calculate hidden deltas
                for (int layerIndex = hiddenLayers.size() - 1; layerIndex >= 0; layerIndex--) {
                    if (layerIndex == hiddenLayers.size() - 1) {
                        flatDeltas = hiddenLayers[layerIndex]->backPropagate(flatDeltas, outLayer->getWeights());
                    }
                    else {
                        flatDeltas = hiddenLayers[layerIndex]->backPropagate(flatDeltas, hiddenLayers[layerIndex + 1]->getWeights());
                    }
                }
                // convert flat deltas to kernel shaped deltas
                matrix lastFlatWeights;
                if (hiddenLayers.empty()) {
                    lastFlatWeights = outLayer->getWeights();
                }
                else {
                    lastFlatWeights = hiddenLayers[0]->getWeights();
                }
                for (int k = 0; k < preKernDeltas.size(); k++) {
                    double total = 0;
                    for (int d = 0; d < flatDeltas.size(); d++) {
                        total += flatDeltas[d] * lastFlatWeights[d][k];
                    }
                    preKernDeltas[k] = total;
                }
                tensor kernDeltas = reshapeVector(preKernDeltas, kernels[kernels.size() - 1]->getNumKernels(), kernels[kernels.size() - 1]->getOutputDim(), kernels[kernels.size() - 1]->getOutputDim());
                // calculate kernel deltas
                for (int kernelIndex = kernels.size() - 1; kernelIndex >= 0; kernelIndex--) {
                    if (kernelIndex != kernels.size() - 1) {
                        kernDeltas = delPadding(kernDeltas, kernels[kernelIndex + 1]->getPadWidth());
                    }
                    kernDeltas = kernels[kernelIndex]->backPropagate(kernDeltas);
                }
                // update layer parameters
                for (auto & kernel : kernels) {
                    kernel->update();
                }
                for (auto & hiddenLayer : hiddenLayers) {
                    hiddenLayer->update();
                }
                outLayer->update();
            }
        }
        *doneTraining = true;
    }

    void miniBatchBackPropagate(vector<tensor> trainData, vector<int> trainLabels, int iterations, int batchSize) {
        *this->progressGoal = iterations * trainData.size();
        // set pre kernel delta size
        vector<double> preKernDeltas(kernels[kernels.size() - 1]->getNumKernels() * pow(kernels[kernels.size() - 1]->getOutputDim(), 2));
        // one-hot encode all labels
        auto labels = oneHotEncode(std::move(trainLabels));
        // for every iteration
        for (int iters = 0; iters < iterations; iters++) {
            // randomize data before creating batches
            vector<int> indexes(trainData.size());
            for (int i = 0; i < trainData.size(); ++i)
                indexes[i] = i;

            std::random_shuffle(indexes.begin(), indexes.end());
            vector<tensor> newData(trainData.size());
            vector<vector<double>> newExpect(trainData.size());
            for (unsigned int i = 0; i < trainData.size(); i++) {
                newData[i] = trainData[indexes[i]];
                newExpect[i] = labels[indexes[i]];
            }
            trainData = newData;
            labels = newExpect;

           // get number of mini-batches
            int numBatches = floor(trainData.size() / batchSize);
            // allocate batch vector
            vector<vector<tensor>> allBatchData(numBatches);
            vector<vector<vector<double>>> allBatchLabels(numBatches);
            // create mini-batches
            for (int b = 0; b < numBatches; b++) {
                vector<tensor> curBatchData(batchSize);
                vector<vector<double>> curBatchLabels(batchSize);
                for (int i = 0; i < batchSize; i++) {
                    curBatchData[i] = trainData[b * batchSize + i];
                    curBatchLabels[i] = labels[b * batchSize + i];
                }
                allBatchData[b] = curBatchData;
                allBatchLabels[b] = curBatchLabels;
            }
            // for every mini-batch
            for (int bIndex = 0; bIndex < numBatches; bIndex++) {
                // for every data point
                for (int dIndex = 0; dIndex < batchSize; dIndex++) {
                    *this->curProgress += 1;
                    vector<double> curLabel = allBatchLabels[bIndex][dIndex];
                    // calculate network output
                    vector<double> result = propagate(allBatchData[bIndex][dIndex]);
                    // calculate output deltas
                    vector<double> flatDeltas = outLayer->miniBatchBackPropagate(curLabel, batchSize);

                    // calculate hidden deltas
                    for (int layerIndex = hiddenLayers.size() - 1; layerIndex >= 0; layerIndex--) {
                        if (layerIndex == hiddenLayers.size() - 1) {
                            flatDeltas = hiddenLayers[layerIndex]->miniBatchBackPropagate(flatDeltas, outLayer->getWeights(), batchSize);
                        } else {
                            flatDeltas = hiddenLayers[layerIndex]->miniBatchBackPropagate(flatDeltas, hiddenLayers[layerIndex +1]->getWeights(), batchSize);
                        }
                    }
                    // convert flat deltas to kernel shaped deltas
                    matrix lastFlatWeights;
                    if (hiddenLayers.empty()) {
                        lastFlatWeights = outLayer->getWeights();
                    }
                    else {
                        lastFlatWeights = hiddenLayers[0]->getWeights();
                    }
                    for (int k = 0; k < preKernDeltas.size(); k++) {
                        double total = 0;
                        for (int d = 0; d < flatDeltas.size(); d++) {
                            total += flatDeltas[d] * lastFlatWeights[d][k];
                        }
                        preKernDeltas[k] = total;
                    }
                    tensor kernDeltas = reshapeVector(preKernDeltas, kernels[kernels.size() - 1]->getNumKernels(),
                                                      kernels[kernels.size() - 1]->getOutputDim(),
                                                      kernels[kernels.size() - 1]->getOutputDim());
                    // calculate kernel deltas
                    for (int kernelIndex = kernels.size() - 1; kernelIndex >= 0; kernelIndex--) {
                        if (kernelIndex != kernels.size() - 1) {
                            kernDeltas = delPadding(kernDeltas, kernels[kernelIndex + 1]->getPadWidth());
                        }
                        kernDeltas = kernels[kernelIndex]->miniBatchBackPropagate(kernDeltas, batchSize);
                    }
                }
                // update layer parameters
                for (auto &kernel: kernels) {
                    kernel->update();
                    kernel->resetDeltas();
                }
                for (auto &hiddenLayer: hiddenLayers) {
                    hiddenLayer->update();
                    hiddenLayer->resetDeltas();
                }
                outLayer->update();
                outLayer->resetDeltas();
            }
        }
        *doneTraining = true;
    }

    void fitModel(vector<tensor> trainData, vector<int> trainLabels, int iterations, int batchSize=1, bool verbose=false) {
        int valSize = floor(trainData.size() / iterations);
        for (int i = 0; i < iterations; i++) {
            // create validation sets for testing purposes
            vector<tensor> curTrainData;
            vector<int> curTrainLabels;
            vector<tensor> valData;
            vector<int> valLabels;
            for (int d = 0; d < trainData.size(); d++) {
                if (d >= valSize * i && d < valSize * i + valSize) {
                    valData.push_back(trainData[d]);
                    valLabels.push_back(trainLabels[d]);
                }
                else {
                    curTrainData.push_back(trainData[d]);
                    curTrainLabels.push_back(trainLabels[d]);
                }
            }

            // set progress bar params
            *curProgress = 0;
            *progressGoal = 1;
            *doneTraining = false;
            if (verbose) { cout << "Training Progress:" << endl; };
            thread thread_obj;
            // stochastic gradient descent thread
            if (batchSize == 1) {
                cout << "SGD" << endl;
                thread_obj = thread(&ConvNet::backPropagate, *this, curTrainData, curTrainLabels, 1);
            }
            // mini-batch gradient descent thread
            else {
                cout << "Mini-Batch" << endl;
                thread_obj = thread(&ConvNet::miniBatchBackPropagate, *this, curTrainData, curTrainLabels, 1, batchSize);
            }
            // progress bar thread
            if (verbose) {
                thread thread_progress(&ConvNet::progressBar, *this);
                thread_progress.join();
            }
            // join threads
            thread_obj.join();
            // validation accuracy
            cout << "Evaluating..." << endl;
            auto res = eval(valData, valLabels) * 100;
            cout << "Validation Accuracy: " << setprecision(4) << res << "%" << endl;

        }
    }

    matrix oneHotEncode(vector<int> labels) {
        int numLabels = outLayer->getNumNeurons();
        matrix res(labels.size());
        for (int lIndex = 0; lIndex < res.size(); lIndex++) {
            vector<double> newVec(numLabels);
            newVec[labels[lIndex]] = 1;
            res[lIndex] = newVec;
        }
        return res;
    }

    double eval(vector<tensor> testData, vector<int> testLabels) {
        double numCorrect = 0;
        for (int d = 0; d < testData.size(); d++) {
            auto res = predict(testData[d]);
            if (res == testLabels[d]) {
                numCorrect++;
            }
        }
        return numCorrect / testData.size();
    }

    void printNet() {
        cout << "Convolutional Neural Network" << endl;
        cout << "____________________________________________________________________" << endl;
        cout << "Convolution Layers: " << endl;
        for (int i = 0; i < kernels.size(); i++) {
            printf("Convolution Layer %d - Kernels: %d\tKernel Dimensions: %d\n", i + 1, kernels[i]->getNumKernels(), kernels[i]->getKernelDim());
        }
        cout << "____________________________________________________________________" << endl;
        cout << "Dense Layers: " << endl;
        for (int i = 0; i < kernels.size(); i++) {
            printf("Dense Layer %d - Neurons: %d\n", i + 1, hiddenLayers[i]->getNumNeurons());
        }
        cout << "____________________________________________________________________" << endl;
        cout << "Output Layer: " << endl;
        printf("Output Layer - Neurons: %d\n", outLayer->getNumNeurons());
        cout << "____________________________________________________________________" << endl;
        printf("Num Connections: %d\n", numConnections);
    }


private:

    // displays a progress bar
    void progressBar() {
        int barWidth = this->barSize;
        HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
        CONSOLE_CURSOR_INFO cursorInfo;
        GetConsoleCursorInfo(out, &cursorInfo);
        cursorInfo.bVisible = false;
        SetConsoleCursorInfo(out, &cursorInfo);
        StopWatch progressWatch;
        progressWatch.reset();
        int startTime = progressWatch.elapsed_time();
        while (!*doneTraining) {
            printBar(*curProgress, *progressGoal, barWidth, progressWatch, startTime);
        }
        printBar(1, 1, barWidth, progressWatch, startTime);
        cout << endl;
        cursorInfo.bVisible = true;
        SetConsoleCursorInfo(out, &cursorInfo);
    }

    // helper for progressBar method
    void printBar(int curVal, int goal, int barWidth, StopWatch watch, int startTime) {
        double progress = double(curVal) / goal;
        cout << "\r[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) cout << "=";
            else if (i == pos) cout << ">";
            else cout << " ";
        }
        cout << "] " << std::setw(3) << int(progress * 100.0) << "% " << loading[int(watch.elapsed_time()) % 4];
        cout.flush();
    }

    matrix convInputs;
    vector<shared_ptr<KernelLayer>> kernels;
    vector<shared_ptr<HiddenLayer>> hiddenLayers;
    shared_ptr<OutputLayer> outLayer;
    int inputChannelCount;
    int inputDim;
    shared_ptr<double> curProgress = make_shared<double>(0.0);
    shared_ptr<double> progressGoal = make_shared<double>(0.0);
    int barSize = 70;
    shared_ptr<bool> doneTraining = make_shared<bool>(false);
    string loading[4] = {" | ", " / ", " - ", " \\ "};
    double lr;
    double m;
    int numConnections = 0;
};
