
from typing import Callable
import tensorflow as tf
import functools
import numpy as np
from .utils import get_weight
from .abstract_class import AbstractPreprocessor


class Preprocessor(AbstractPreprocessor):
    """this class crops deterministic patches using convolution"""
    def __init__(self, training, num_patches, patch_size, classes):
        super().__init__(training, num_patches, patch_size, classes)

    @staticmethod
    def _get_patches_by_convolution(combined, h, w, num_patches, patch_size):
        """
        for a given number of patches to be extracted, this method first finds the stride
        size so that a convolutional patch extraction yields the desired number of patches.

        For convolutional filters, the output size is determined by:
        num_filter = ((input_size-filter_size)/stride)+1

        Hence, the stride size for given number of filters (patches) is as follows:
        s = (input_size - filter_size) / (num_filters - 1)

        The determined stride size is then used in tf.image.extract_patches to crop the patches.
        This function must be executed in eager mode, since the determined stride size may not 
        be a tensor but an integer

        args:
        - combined: the concatentation of the input image and its otsu binarization
        - h: height of the input image
        - w: width of the input image
        """
        # determine the factors of num_patches 
        factors = functools.reduce(list.__add__, ([(tf.constant(i), num_patches//i)] for i in range(
            1, int(np.sqrt(num_patches))+1) if num_patches % i == 0))
        # determine the aspect ratio of the image
        aspect_ratio = tf.math.divide(tf.math.minimum(h, w), tf.math.maximum(h, w))
        # choose the factors, which ratio is closest to the aspect ratio of the image
        # For example, if 4 images of a square image have to be extracted, we want to
        # extract 2 images per "column" instead of all patches lying in a line
        idx = tf.argmin([tf.abs(tf.subtract(tf.divide(*factors[i]), aspect_ratio)) for i in range(0, len(factors), 1)])
        patch1, patch2 = tf.unstack(tf.gather(tf.stack(factors), idx))
        # extract more patches from the larger side of the image
        if tf.less(h,w):
            num_patches_h, num_patches_w = tf.minimum(patch1, patch2), tf.maximum(patch1, patch2)
        else:
            num_patches_h, num_patches_w = tf.maximum(patch1, patch2), tf.minimum(patch1, patch2)
        # determine stride size according to formula given in description
        num_patches_h, num_patches_w = tf.subtract(num_patches_h, 1), tf.subtract(num_patches_w, 1)
        stride_h = tf.cast(tf.math.floor(tf.divide(tf.cast(tf.subtract(h, patch_size), tf.int64), tf.cast(num_patches_h, tf.int64))), tf.int16)
        stride_w = tf.cast(tf.math.floor(tf.divide(tf.cast(tf.subtract(w, patch_size), tf.int64), tf.cast(num_patches_w, tf.int64))), tf.int16)
        # extract and return patches
        patches = tf.reshape(
            tf.image.extract_patches(
                images=tf.expand_dims(combined, 0),
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, stride_h.numpy(), stride_w.numpy(), 1],
                rates=[1, 1, 1, 1],
                padding='VALID'), 
            (-1, patch_size, patch_size, 4))

        return patches

    def _get_patches_and_weights(self, combined, height, width):
        patches = tf.py_function(
            self._get_patches_by_convolution, 
            inp=[combined, height, width, self.num_patches, self.patch_size], 
            Tout=tf.float32)
        img_patches = tf.cast(patches[:,:,:,:3], tf.float32)
        # get weight of patches
        otsu = tf.cast(patches[:,:,:,-1], tf.int32)
        weights = tf.map_fn(lambda x: get_weight(x), otsu, fn_output_signature=tf.float64)
        return img_patches, weights

    def preprocess_for_train(   
        self,
        combined: tf.Tensor, 
        height: tf.Tensor, 
        width: tf.Tensor,
        **_):
        """crop deterministic patches by convolution"""        
        return self._get_patches_and_weights(combined, height, width)

    def preprocess_for_validation(   
        self,
        combined: tf.Tensor, 
        height: tf.Tensor, 
        width: tf.Tensor,
        **_):
        """crop deterministic patches by convolution"""
        return self._get_patches_and_weights(combined, height, width)
