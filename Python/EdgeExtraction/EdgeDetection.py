import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ctypes import cdll, c_int
import ctypes
from pathlib import Path


lib = cdll.LoadLibrary(f'{Path.cwd()}/EdgeExtraction.dll')


# EdgeExtractor Class
class EdgeExtractor:
    # constructor
    def __init__(self, filename):
        self.filename = filename
        # open image and convert to grayscale numpy array
        self.gray_px = np.array(Image.open(f'{self.filename}').convert("L").__array__(), dtype=int)
        # get image shape, then pad image and get padded shape
        self.im_len = len(self.gray_px)
        self.im_width = len(self.gray_px[0])
        self.padded = np.pad(self.gray_px, pad_width=2, constant_values=0)
        self.padded_len = len(self.padded)
        self.padded_width = len(self.padded[0])

    # extract edges from image
    def extract_edges(self):
        # flatten the padded image
        flat_padded = self.padded.flatten()
        # convert the flattened image into a C array
        c_px = flat_padded.ctypes.data_as(ctypes.POINTER(c_int))
        # define types for extractEdges C++ method
        lib.extractEdges.argtypes = [ctypes.POINTER(c_int), c_int, c_int]
        lib.extractEdges.restype = ctypes.POINTER(c_int)
        # pass padded array to C++ method
        res_ptr = lib.extractEdges(c_px, self.padded_width, self.padded_len)
        # convert result to a numpy array
        res_array = np.ctypeslib.as_array(res_ptr, shape=(self.im_len * self.im_width,)).reshape([len(self.gray_px), len(self.gray_px[0])])
        return res_array

    def saveOutput(self):
        plt.savefig(f'{self.filename[:len(self.filename) - 4]}_edge.jpg')


def main():
    filename = input('Enter file name: ')
    edges = EdgeExtractor(filename)
    plt.imshow(edges.extract_edges())
    plt.show()


if __name__ == '__main__':
    main()
