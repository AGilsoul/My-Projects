#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <random>

using namespace std;


/**Point struct definition, each one represents a data point**/

struct point {
    //Point constructor
    point(string label, vector<double> dblData): label(label), dblData(dblData) {}
    //Point label
    string label;
    //Double data of each category for point
    vector<double> dblData;

};


/**Function declarations**/

vector<point> readFile();
vector<double> sortVector(vector<double>);
template <typename type>
void printVector(vector<type>);
vector<double> normalize(vector<double>);
vector<vector<double>> getDistances(vector<point>, vector<point>);
template <typename type>
vector<type> vectorSplit(vector<type>, int, int);
template <typename type>
bool inVector(type, vector<type>);
template <typename type>
int getIndex(type, vector<type>);
double runTest(vector<point>, vector<point>, int k);
void applyWeights(vector<double>&, vector<int>, double);


/**Main**/

int main()
{
    //Variable declarations and definitions
    double accuracy = 0;
    double result;
    int iterations, k;

    cout << "Enter number of nearest points/neighbors (K) to analyze (recommended is 22): ";
    cin >> k;
    cout << "Enter number of epochs(iterations) to test data on: ";
    cin >> iterations;

    //Initialize random number generator
    auto rng = default_random_engine {};
    //Initialize patientData point vector from readFile() function
    vector<point> patientData = readFile();

    //Repeats tests based on number of epochs/iterations
    for (unsigned int b = 1; b <= iterations; b++) {
        //Randomly shuffle patientData based on random number generator
        shuffle(begin(patientData), end(patientData), rng);
        //Creates dataList array, which stores a vector of each sample point's double data, same size as dblData vectors in point struct
        vector<double> dataList[patientData[0].dblData.size()];

        //Loops through each vector in dataList array, copying data from each point to the corresponding array indexes
        for (int x = 0; x < sizeof(dataList) / sizeof(dataList[0]); x++) {
            for (int patient = 0; patient < patientData.size(); patient++) {
                dataList[x].push_back(patientData[patient].dblData[x]);
            }
            //Normalizes values in each vector in dataList
            dataList[x] = normalize(dataList[x]);
        }

        //copies the normalized data back onto the original patientData points
        for (int cat = 0; cat < sizeof(dataList) / sizeof(dataList)[0]; cat++) {
            for (int i = 0; i < dataList[cat].size(); i++) {
                patientData[i].dblData[cat] = dataList[cat][i];
            }
        }

        //Splits patient data into two vectors, training data and test data
        auto trainSplit = vectorSplit(patientData, 0, ceil(patientData.size() * 0.95));
        auto testSplit = vectorSplit(patientData, ceil(patientData.size() * 0.95), patientData.size() - 1);

        //Gets accuracy percentage results from runTest() function
        result = runTest(testSplit, trainSplit, k);
        //adds result to total accuracy to get average
        accuracy += result;

        cout << "Accuracy of epoch " << b << ": " << result << "%" << endl;

    }
    cout << endl << "Percent of correctly identified tumors (Malignant/Benign): " << accuracy / iterations << "%" << endl;



    return 0;
}


/** Function Definitions **/

//Applies weights to a data vector, for each index in colMod, by multiplying the values by the weight parameter
//Not currently used
void applyWeights(vector<double> &data, vector<int> colMod, double weight) {
    for (int col = 0; col < colMod.size();  col++) {
        data[colMod[col]] *= weight;
    }
}

//Returns a double of the correct percent guesses
double runTest(vector<point> testData, vector<point> practiceData, int k) {
    //gets distances of each testData point to all practiceData points
    auto distances = getDistances(testData, practiceData);
    double correctGuesses = 0;
    //Loops through each testData point
    for (int tp = 0; tp < distances.size(); tp++) {
        vector<double> sortedDistances;
        vector<point> sortedPoints;
        sortedPoints.push_back(practiceData[0]);
        sortedDistances.push_back(distances[tp][0]);
        vector<string> guesses;
        vector<double> guessCounts;
        string guess;
        //Sorts distances of practiceData points to testData point through an insertion sort
        //Sorts the points and their labels accordingly as well
        for (int p = 0; p < distances[tp].size(); p++) {
            for (int d = 0; d < sortedDistances.size(); d++) {
                if (distances[tp][p] < sortedDistances[d]) {
                    sortedDistances.insert(sortedDistances.begin() + d, distances[tp][p]);
                    sortedPoints.insert(sortedPoints.begin() + d, practiceData[p]);
                    break;
                }
                else if (d == sortedDistances.size() - 1) {
                    sortedDistances.push_back(distances[tp][p]);
                    sortedPoints.push_back(practiceData[p]);
                    break;
                }
            }
        }
        //Takes the k-nearest-neighbors to test point
        //Makes prediction based on which category has the most labeled points
        //Ff the result is even, predicts based on the closest point
        for (int i = k; i >= 0; i--) {
            int it = k;
            if (!inVector(sortedPoints[i].label, guesses)) {
                guesses.push_back(sortedPoints[i].label);
                if (sortedPoints[i].label == "M") {
                    guessCounts.push_back(2);
                }
                else {
                    guessCounts.push_back(1);
                }
            }
            else {
                if (sortedPoints[i].label == "M") {
                    guessCounts[getIndex(sortedPoints[i].label, guesses)] += 2;
                }
                else {
                    guessCounts[getIndex(sortedPoints[i].label, guesses)] += 1;
                }
            }
        }
        double guessMax = 0;
        //Determines which label has more instances
        for (int i = 0; i < guessCounts.size(); i++) {
            if (guessCounts[i] > guessMax) {
                guessMax = guessCounts[i];
            }
        }
        guess = guesses[getIndex(guessMax, guessCounts)];
        //Checks guess
        if (guess == testData[tp].label) {
            correctGuesses++;
        }
    }
    //Returns percent correct of guesses
    return correctGuesses / testData.size() * 100;
}

//Gets the index position of an item in a vector
//If the value is not present, returns -1
template <typename type>
int getIndex(type item, vector<type> data) {
    for (int i = 0; i < data.size(); i++) {
        if (item == data[i]) {
            return i;
        }
    }
    return -1;
}

//Prints the contents of a vector to the console
template <typename type>
void printVector(vector<type> data) {
    for (int i = 0; i < data.size() - 1; i++) {
        cout << data[i] << " ";
    }
    cout << data[data.size() - 1];
}

//Returns true if an item is in a vector, false if not
template <typename type>
bool inVector(type item, vector<type> data) {
    for (int i = 0; i < data.size(); i++) {
        if (data[i] == item) {
            return true;
        }
    }
    return false;
}

//Insertion sort for a vector of doubles, returns sorted vector
vector<double> sortVector(vector<double> data) {
    vector<double> sortedData;
    sortedData.push_back(data[0]);
    for (int i = 1; i < data.size(); i++) {
        for (int x = 0; x < sortedData.size(); x++) {
            if (data[i] < sortedData[x]) {
                sortedData.insert(sortedData.begin() + x, data[i]);
                break;
            }
            else if (x == sortedData.size() - 1) {
                sortedData.push_back(data[i]);
                break;
            }
        }
    }
    return sortedData;
}

//Normalizes a vector of doubles to a have a mean of 0 and a standard deviation of 1 in both directions
vector<double> normalize(vector<double> data) {
    auto sortedData = sortVector(data);
    double median = sortedData[ceil(sortedData.size() / 2)];
    double p25 = sortedData[ceil(sortedData.size() / 4)];
    double p75 = sortedData[ceil(sortedData.size() / 2) + ceil(sortedData.size() / 4)];
    double dataMin = sortedData[0];
    double dataMax = sortedData[sortedData.size() - 1];
    for (int x = 0; x < data.size(); x++) {
        data[x] = (data[x] - median) / (p75 - p25);
    }
    return data;
}

//Creates a new vector as a fragment of a larger vector
template <typename type>
vector<type> vectorSplit(vector<type> data ,int start, int fin) {
    vector<type> newVec;
    for (int i = start; i <= fin; i++) {
        newVec.push_back(data[i]);
    }
    return newVec;
}

//Returns a vector of distances for each point in the test vector to all of the points in the points vector
//Takes euclidean distance of the different data items in the double data vectors in the point structs
//Euclidean distance: square root((x0 - x1)^2 + (y0 - y1)^2 + ...)
vector<vector<double>> getDistances(vector<point> test, vector<point> points) {
    vector<vector<double>> testPoints;
    for (int tp = 0; tp < test.size(); tp++) {
        vector<double> distances;
        for (int p = 0; p < points.size(); p++) {
            double total = 0;
            for (int l = 0; l < points[p].dblData.size(); l++) {
                total += pow((points[p].dblData[l] - test[tp].dblData[l]), 2);

            }
            distances.push_back(sqrt(total));
        }
        testPoints.push_back(distances);
    }

    return testPoints;
}

//Reads a csv file and assigns data to strings
//The diagnosis string is used in the point constructor to label the point
//The rest of the data is included in the dblData vector in each point
//Returns a vector of all the newly created points
vector<point> readFile() {
    vector<point> points;
    //strings to be used for reference and assignment of values when reading the file and assigning to the string list sList
    string id, diagnosis, radMean, texMean, perMean, areaMean, smoothMean, compMean, concMean, concPointMean, symMean, fracMean, radSE, texSE, perSE, areaSE, smoothSE, compSE, concSE, concPointSE, symSE, fracSE, radWorst, perWorst, areaWorst, smoothWorst, compWorst, concWorst, concPointWorst, symWorst, fracWorst;
    string sList[] = {id, diagnosis, radMean, texMean, perMean, areaMean, smoothMean, compMean, concMean, concPointMean, symMean, fracMean, radSE, texSE, perSE, areaSE, smoothSE, compSE, concSE, concPointSE, symSE, fracSE, radWorst, perWorst, areaWorst, smoothWorst, compWorst, concWorst, concPointWorst, symWorst, fracWorst};
    //Reads from the file "Breast_Cancer.csv"
    ifstream fin("Breast_Cancer.csv", ios::in);
    while (!fin.eof()) {
        vector<double> dData;
        for (int i = 0; i < sizeof(sList) / sizeof(sList[0]); i++) {
            if (i != sizeof(sList) / sizeof(sList[0]) - 1) {
                getline(fin, sList[i], ',');
            }
            else {
                getline(fin, sList[i], '\n');
            }
            if (i != 1 && i != 0) {
                dData.push_back(stod(sList[i]));
            }
        }

        point newPoint(sList[1], dData);
        points.push_back(newPoint);
    }
    fin.close();
    return points;
}
