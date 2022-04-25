#include <iostream>
#include "equations.h"

int main() {

    polynomial p1({3, -1, 1}, {-2, 0, 5}, "x");

    polynomial p2({-1, 1, 3}, {0, 1, 5}, "x");
    cout << p1 << " * " << p2 << endl;
    polynomial result = p1 * p2;
    cout << result << endl << endl;


    return 0;
}
