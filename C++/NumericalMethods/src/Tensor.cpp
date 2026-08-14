#include "Tensor.h"

using std::vector;
using std::invalid_argument;


// FIX ALL THIS

namespace NumericalMethods
{
    template <typename T, size_t Rank>
    Tensor<T, Rank>::Tensor(vector<int> dims): dims(dims)
    {
        // Make sure we receive the proper amount of dimensions.
        if (dims.size() != Rank)
        {
            throw invalid_argument("Tensor rank must equal number of dimensions minus 1.");
        }

        // If the rank is greater than 0, fill the references to the sub tensors.
        if constexpr (Rank > 0)
        {
            this->subTensors = vector<Tensor<T, Rank-1>>;
            vector<int> newDims(dims.begin() + 1, dims.end());

            for (size_t i = 0; i < dims[0]; i++)
            {
                this->subTensors.push_back(Tensor<T, Rank-1>(newDims));
            }
        }
    }

    template <typename T, size_t Rank>
    auto& Tensor<T, Rank>::operator[](size_t idx) const
    {
        if constexpr (Rank == 0)
        {
            return this->value;
        }
        else
        {
            return subTensors[idx];
        }
    }

    template <typename T, size_t Rank>
    auto Tensor<T, Rank>::operator[](size_t idx)
    {
        if constexpr (Rank == 0)
        {
            return this->value;
        }
        else
        {
            return subTensors[idx];
        }
    }
}