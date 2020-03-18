import numpy as np


class MyConv:
    def __init__(self, image):
        self.shape = image.shape
        self.image = image

    def convolution(self, filler, bias, stride=1):
        """
        :param self: image file to convolute around
        :param filler: filter, to avoid built-in name filter
        :param bias: exactly that. bias
        :param stride: stride size (default at 1)
        :return: convoluted image

        Convolves will filter over the image using stride of size `stride`.
        """

        (n_f, n_c_f, f, _) = filler.shape  # dimension of filter
        n_c, in_dim, _ = self.shape  # dimension of shape

        out_dim = int((in_dim - f) / stride) + 1  # dimension of output

        # Check if the dimensions match
        assert n_c == n_c_f, "Dimensions of filter must match input image dimension"

        output = np.zeros((n_f, out_dim, out_dim))  # Placeholder of convoluted image

        # Beginning of convolution here
        for curr_f in range(n_f):
            curr_y = out_y = 0
            while curr_y + f <= in_dim:
                curr_x = out_x = 0
                while curr_x + f <= in_dim:
                    output[curr_f, out_y, out_x] = \
                        np.sum(filler[curr_f] * self[:, curr_y:curr_y + f, curr_x:curr_x + f]) + bias[curr_f]
                    curr_x += stride
                    out_x += 1
                curr_y += stride
                out_y += 1

        return output

    def max_pooling(self, f=2, stride=2):
        """
        :param self: image file to downsample
        :param f: kernel size
        :param stride: stride size
        :return:
        """

        n_c, h_prev, w_prev = self.shape

        h = int((h_prev - f) / stride) + 1
        w = int((w_prev - f) / stride) + 1

        downsampled = np.zeros((n_c, h, w))

        for i in range(n_c):
            curr_y = out_y = 0
            while curr_y + f <= h_prev:
                curr_x = out_x = 0
                while curr_x + f <= w_prev:
                    downsampled[i, out_y, out_x] = np.max(self[i, curr_y:curr_y + f, curr_x:curr_x + f])
                    curr_x += stride
                    out_x += 1
                curr_y += stride
                out_y += 1

        return downsampled
