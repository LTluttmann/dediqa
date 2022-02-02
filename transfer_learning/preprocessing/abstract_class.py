from abc import ABC, abstractmethod
from typing import Tuple
import tensorflow as tf
from . import utils as utils 

class AbstractPreprocessor(ABC):
    """Abstract class providing interface for preprocessor classes"""
    def __init__(self, training, num_patches, patch_size, classes):
        super().__init__()
        self.training = training
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.classes = classes

    @abstractmethod
    def preprocess_for_train(
        self, image, otsu, label, height, width, **kwargs
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        """function that implements the tf.data pipeline for the training data"""
        ...
    
    @abstractmethod
    def preprocess_for_validation(
        self, image, otsu, label, height, width, **kwargs
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        """function that implements the tf.data pipeline for the validation / test data"""
        ...

    def preprocess_image(
        self, image, otsu, label, height, width, resize:str=None, **kwargs
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """calls the preprocessing function for training or for validation
         Args:
        - image: one single image to be processed
        - otsu: otsu binarization of image
        - label: label specifying DPI of input image. Will be the same for each patch
        - height: species the height of the image (not the resulting patches)
        - width: species the width of the image. Note: height and width have to be parsed as they cannot be
          extracted from image tensor during graph runtime.
        Returns:
        - patches: a tensor containing 'self.num_patches' crops of the original image
        - labels: a tensor containing the label for each patch (which is the label of the image)
        - weights: a tensor containing the weight for each patch for the loss function
        - resize: string specifying the resize method: min, max or random
        """
        print(f"preprocces images in {'training' if self.training else 'test'} mode")
        # determine rank of label (1 if it is a list, 0 if it is a scalar)
        label_rank = 1 if self.classes is not None else 0
        # make list with label -> assign each crop the label of the image
        labels = tf.tile(tf.expand_dims(label, 0), tf.constant([self.num_patches, label_rank]))
        
        # concat original image and otsu to ensure same patches are cropped
        img_w_binary = tf.concat([image, otsu], axis=-1)
        # call the specified resize function or perform minimum resizing to avoid errors
        resize_func = f"{resize}_resize"
        if hasattr(utils, resize_func):
            func = getattr(utils, resize_func)
            img_w_binary = func(
                img_w_binary, height, width, label, 
                patch_size=self.patch_size,
                classes=self.classes, 
                num_patches=self.num_patches
            )
        else:
            img_w_binary = utils.min_resize(
                img_w_binary, height, width, 
                patch_size=self.patch_size, 
                num_patches=self.num_patches
            )
        # call the preprocess function
        if self.training:
            patches, weights = self.preprocess_for_train(img_w_binary, height, width, **kwargs)
        else:
            patches, weights =  self.preprocess_for_validation(img_w_binary, height, width, **kwargs)
        # return patches, labels and weights -> order is important here
        return patches, labels, weights