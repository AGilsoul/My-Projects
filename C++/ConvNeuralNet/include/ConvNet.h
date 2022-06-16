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

class ConvNet {
public:
    ConvNet(int inputChannelCount, int inputDim, double lr, double m=0.0, bool verbose=false): inputChannelCount(inputChannelCount), inputDim(inputDim), lr(lr), m(m), verbose(verbose) {}

    void addKernel(int numKernels, int kernelDim, bool padding=true, bool pooling=false, const string& poolType="avg") {
        if (kernels.empty()) {
            kernels.push_back(make_shared<KernelLayer>(numKernels, kernelDim, inputChannelCount, inputDim, lr, m, padding, pooling, poolType));
        }
        else {
            kernels.push_back(make_shared<KernelLayer>(numKernels, kernelDim, kernels[kernels.size() - 1]->getNumChannels(), kernels[kernels.size() - 1]->getOutputDim(), lr, m, padding, pooling, poolType));
        }
    }

    void addHiddenLayer(int numNeurons) {
        if (kernels.empty()) {
            throw exception("Add kernels before dense layers");
        }

        if (hiddenLayers.empty()) {
            int numInputs = kernels[kernels.size() - 1]->getNumChannels() * pow(kernels[kernels.size() - 1]->getOutputDim(), 2);
            hiddenLayers.push_back(make_shared<HiddenLayer>(numNeurons, numInputs, lr, m));
        }
        else {
            int numInputs = hiddenLayers[hiddenLayers.size() - 1]->getNumNeurons();
            hiddenLayers.push_back(make_shared<HiddenLayer>(numNeurons, numInputs, lr, m));
        }
    }

    void addOutputLayer(int numNeurons) {
        if (kernels.empty()) {
            throw exception("Add kernels before dense layers");
        }

        if (hiddenLayers.empty()) {
            int numInputs = kernels[kernels.size() - 1]->getNumChannels() * pow(kernels[kernels.size() - 1]->getOutputDim(), 2);
            outLayer = make_shared<OutputLayer>(numNeurons, numInputs, lr, m);
        }
        else {
            int numInputs = hiddenLayers[hiddenLayers.size() - 1]->getNumNeurons();
            outLayer = make_shared<OutputLayer>(numNeurons, numInputs, lr, m);
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

    void backPropagate(vector<tensor> trainData, vector<int> trainLabels, int iterations, vector<tensor> valData={{{{}}}}, vector<int> valLabels={}) {
        *this->progressGoal = iterations * trainData.size();
        // set pre kernel delta size
        vector<double> preKernDeltas(kernels[kernels.size() - 1]->getNumChannels() * pow(kernels[kernels.size() - 1]->getOutputDim(), 2));
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
                tensor kernDeltas = reshapeVector(preKernDeltas, kernels[kernels.size() - 1]->getNumChannels(), kernels[kernels.size() - 1]->getOutputDim(), kernels[kernels.size() - 1]->getOutputDim());
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

    void fitModel(vector<tensor> trainData, vector<int> trainLabels, int iterations, vector<tensor> valData={{{{}}}}, vector<int> valLabels={}) {
        *curProgress = 0;
        *progressGoal = 1;
        *doneTraining = false;
        if (this->verbose) { cout << "Training Progress:" << endl; };
        thread thread_obj(&ConvNet::backPropagate, *this, trainData, trainLabels, iterations, valData, valLabels);
        if (this->verbose) {
            thread thread_progress(&ConvNet::progressBar, *this);
            thread_progress.join();
        }
        thread_obj.join();
        cout << "Evaluating..." << endl;
        cout << "Accuracy: " << setprecision(4) << eval(valData, valLabels) * 100 << "%" << endl;
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
            if (predict(testData[d]) == testLabels[d]) {
                numCorrect++;
            }
        }
        return numCorrect / testData.size();
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
    bool verbose;
    double lr;
    double m;
};
