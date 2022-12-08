import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# EdgeExtractor Class
class EdgeExtractor:
    # vertical edge filter
    vert_kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])
    # horizontal edge filter
    horz_kernel = vert_kernel.T

    # constructor
    def __init__(self, filename):
        # open image and convert to grayscale numpy array
        img = Image.open(f'{filename}')
        gray_img = img.convert("L")
        self.gray_px = np.array(gray_img)
        # extract horizontal and vertical edges, then compose them
        self.vert_edges = self.extract_edges(self.vert_kernel)
        self.horz_edges = self.extract_edges(self.horz_kernel)
        self.all_edges = self.compose_edges()
        # display and save image
        plt.imshow(self.all_edges)
        plt.savefig(f'{filename[:len(filename) - 4]}_edge.jpg')

    # extract edges from image with given filter
    def extract_edges(self, cur_filter):
        # pad grayscale numpy array with 0s
        new_pix = np.pad(self.gray_px, pad_width=2, constant_values=0)
        # create empty numpy array to store edge values
        edges = np.empty(self.gray_px.shape, dtype=int)
        # filter over image
        for x in range(len(self.gray_px)):
            for y in range(len(self.gray_px[0])):
                for i_x in range(3):
                    for (i_y) in range(3):
                        edges[x][y] += new_pix[x + i_x][y + i_y] * cur_filter[i_x][i_y]
        # return the absolute value of each pixel
        return abs(edges)

    # combine horizontal and vertical edges
    def compose_edges(self):
        # create empty numpy array to store results
        edges = np.empty(self.gray_px.shape, dtype=int)
        # add values of pixels for both types of edges
        for x in range(len(self.gray_px)):
            for y in range(len(self.gray_px[0])):
                edges[x][y] = self.vert_edges[x][y] + self.horz_edges[x][y]
        return edges


def main():
    filename = input('Enter file name: ')
    EdgeExtractor(filename)
    plt.show()


if __name__ == '__main__':
    main()
