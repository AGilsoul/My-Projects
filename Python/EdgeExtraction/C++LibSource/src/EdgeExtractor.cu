//
// Created by agils on 12/8/2022.
//

#include "../include/EdgeExtractor.cuh"

// apply horizontal and vertical edge filters to image
__global__ void applyFilter(int* im, int* horzFilter, int* vertFilter, int* imOut, int outW, int imW) {
    int xVal = threadIdx.x;
    int yVal = blockIdx.x;

    // get horizontal edge output pixel
    // (if both applied at the same time, not as sharp)
    int horSum = 0;
    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            horSum += im[(xVal + x) * imW + yVal + y] * horzFilter[x * 3 + y];
        }
    }
    horSum = abs(horSum);
    // get vertical edge output pixel
    int vertSum = 0;
    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            vertSum += im[(xVal + x) * imW + yVal + y] * vertFilter[x * 3 + y];
        }
    }
    vertSum = abs(vertSum);
    imOut[xVal * outW + yVal] = vertSum + horSum;
}

DELLEXPORT int* extractEdges(int* imageArray, int width, int length) {
    // output image shape
    int outW = width - 4;
    int outL = length - 4;
    // number of output elements
    int numOutElems = outW * outL;
    // number of input image elements
    int numImElems = width * length;

    // define and instantiate host and device arrays
    int *kernImage, *kernOutput, *hostOutput, *vertKernel, *horzKernel;
    cudaMalloc((void**)&kernImage, sizeof(int) * numImElems);
    cudaMalloc((void**)&kernOutput, sizeof(int) * numOutElems);
    cudaMalloc((void**)&vertKernel, sizeof(int) * 9);
    cudaMalloc((void**)&horzKernel, sizeof(int) * 9);
    hostOutput = (int*)malloc(sizeof(int) * numOutElems);
    // vertical and horizontal filters
    int tempVert[9] = { 1, 0, -1, 1, 0, -1, 1, 0, -1};
    int tempHorz[9] = { 1, 1, 1, 0, 0, 0, -1, -1, -1};

    // copy host arrays to device
    cudaMemcpy(vertKernel, tempVert, sizeof(int) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(horzKernel, tempHorz, sizeof(int) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(kernImage, imageArray, sizeof(int) * numOutElems, cudaMemcpyHostToDevice);
    // apply filters to input image
    applyFilter<<<outW, outL>>>(kernImage, horzKernel, vertKernel, kernOutput, outW, width);
    // copy result device array to host array
    cudaMemcpy(hostOutput, kernOutput, sizeof(int) * numOutElems, cudaMemcpyDeviceToHost);

    // free device arrays
    cudaFree(kernImage);
    cudaFree(kernOutput);
    cudaFree(vertKernel);
    cudaFree(horzKernel);

    return hostOutput;
}




