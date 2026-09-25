#include "../include/Tensor.h"
#include <vector>

using std::vector;
using NumericalMethods::Tensor;

int main() {

    vector<int> vec = {2, 3};
    Tensor<double, 2> myTensor(vec);

    return 0;
}