#include <iostream>
#include "../include/csvTable.h"

int main() {
    csvTable table("../res/test.csv");
    table.printTable();
    table.printColumnInfo(0);
    return 0;
}
