import tensorflow as tf
from .utils import get_weight
from .abstract_class import AbstractPreprocessor

class Preprocessor(AbstractPreprocessor):
    """this class crops random patches from an image and weights the according to 
    the entropy of the corresponding otsu binarization
    """
    def __init__(self, *args):
        super().__init__(*args)

    @staticmethod
    def _crop_random(combined, num_patches, patch_size):
        """Get `num_patches` random crops from the image"""
        # init lists containing the patches
        patches, weights = [], []
        for _ in range(num_patches):
            patch = tf.image.random_crop(combined, [patch_size, patch_size, 4])
            patches.append(tf.cast(patch[:,:,:3], tf.float32))
            otsu_patch = patch[:,:,-1]
            weights.append(get_weight(otsu_patch))
        patches = tf.stack(patches)
        weights = tf.stack(weights)

        return patches, weights

    @staticmethod
    def _crop_random_stateless(combined, num_patches, patch_size):
        """Get `num_patches` random crops from the image. Crops
        patches randomly, but does yield the same crops each time
        """
        # init lists containing the patches
        patches, weights = [], []
        for i in range(num_patches):
            seed = (i, 0)
            patch = tf.image.stateless_random_crop(
                combined, [patch_size, patch_size, 4], seed=seed
            )
            patches.append(tf.cast(patch[:,:,:3], tf.float32))
            otsu_patch = patch[:,:,-1]
            weights.append(get_weight(otsu_patch))
        patches = tf.stack(patches)
        weights = tf.stack(weights)

        return patches, weights

    def preprocess_for_train(
        self,
        combined: tf.Tensor, 
        *__,
        **_):
        """Get `self.num_patches` random crops from the image"""
        print(f"the following arguments have not been used for training: {_}")
        return self._crop_random(combined, self.num_patches, self.patch_size)

    def preprocess_for_validation(
        self,
        combined: tf.Tensor, 
        *__,
        **_):
        """Get `self.num_patches` random crops from the image"""
        print(f"the following arguments have not been used for validation/test: {_}")
        return self._crop_random_stateless(combined, self.num_patches, self.patch_size)