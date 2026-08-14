#pragma once

#include <iostream>
#include <vector>

using std::ostream;
using std::cout;
using std::endl;
using std::vector;
using std::size_t;

namespace NumericalMethods
{
    /**
     * @brief Represents a Tensor of rank [Rank].
     */
    template <typename T, size_t Rank> class Tensor 
    {
    public:
        vector<int> dims;

        Tensor(vector<int> dims);
        
        auto& operator [](size_t idx) const;
        auto operator [](size_t idx);

    private:
        vector<Tensor*> subTensors;
        T value = 0;

    };
}
