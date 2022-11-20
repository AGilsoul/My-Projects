#include <iostream>
#include "../include/csvReader.h"


int main() {
    csvReader csv("../res/test.csv");
    cout << csv << endl;

    return 0;
}
