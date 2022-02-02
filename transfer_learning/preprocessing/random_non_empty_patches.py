import tensorflow as tf
from .utils import tf_entropy
from .abstract_class import AbstractPreprocessor

class Preprocessor(AbstractPreprocessor):
    def __init__(self, training, num_patches, patch_size, classes):
        super().__init__(training, num_patches, patch_size, classes)
    
    @staticmethod
    def _crop_random(combined, array, idx, patch_size):
        """crops random image and passes it to the array if the entropy of the corresponding
        otsu binarized patch is greater than zero. If it is zero, nothing happens
        """
        patch = tf.image.random_crop(combined, [patch_size, patch_size, 4])
        otsu_patch = patch[:,:,-1]
        bins = tf.math.bincount(tf.cast(otsu_patch, tf.int32), minlength=2, maxlength=2)
        entropy = tf_entropy(bins)
        # write the image patch to the tensor only if the entropy is larger than zero
        array, idx = tf.cond(
            tf.greater(entropy, 0), 
            lambda: [array.write(idx, tf.cast(patch[:,:,:3], tf.float32)), tf.add(idx, 1)], 
            lambda: [array, idx]
        )
        return array, idx

    @staticmethod
    def _crop_random_stateless(combined, array, idx, patch_size):
        """performs the same operation as '_crop_random', but uses a seed to guarantee the same 
        crops each time the function is called. This is useful for the validation set, where 
        the effect of randomness has to be reduced to an absolute minimum. 
        """
        # setting the seed
        seed = (idx, 0)
        # cropping a patch
        patch = tf.image.stateless_random_contrast(
            combined, [patch_size, patch_size, 4], seed=seed
        )
        # determine otsu entropy
        otsu_patch = patch[:,:,-1]
        bins = tf.math.bincount(tf.cast(otsu_patch, tf.int32), minlength=2, maxlength=2)
        entropy = tf_entropy(bins)
        # write the image patch to the tensor only if the entropy is larger than zero
        array, idx = tf.cond(
            tf.greater(entropy, 0), 
            lambda: [array.write(idx, tf.cast(patch[:,:,:3], tf.float32)), tf.add(idx, 1)], 
            lambda: [array, idx]
        )
        return array, idx

    def preprocess_for_train(   
        self,
        combined: tf.Tensor, 
        *__,
        **_):
        """crop random patches. Patches that do not contain any content are discarded
        so that it is guaranteed, that there are 'num_patches' meaningful training examples
        """
        # initialize empty tensor where the patches can be passed to
        ta = tf.TensorArray(tf.uint8, size=self.num_patches, element_shape=(self.patch_size, self.patch_size, 3))
        # condition for the while loop: index must be less than the number of patches (index starts with zero)
        condition = lambda ta, i: tf.less(i, self.num_patches)
        # perform operation as long 'num_patches' patches have been written to the tensor array
        ta, _ = tf.while_loop(
            condition, 
            lambda ta, i: self._crop_random(combined, ta, i, self.patch_size), 
            loop_vars = [ta, 0]
        )
        ta = ta.stack()
        # no weighting 
        weights = tf.ones(self.num_patches)
        return ta, weights

    def preprocess_for_validation(   
        self,
        combined: tf.Tensor, 
        *__,
        **_):
        """crop random patches. Patches that do not contain any content are discarded
        so that it is guaranteed, that there are 'num_patches' meaningful training examples
        """
        ta = tf.TensorArray(tf.uint8, size=self.num_patches, element_shape=(self.patch_size, self.patch_size, 3))
        condition = lambda ta, i: tf.less(i, self.num_patches)
        ta, _ = tf.while_loop(
            condition, 
            lambda ta, i: self._crop_random_stateless(combined, ta, i, self.patch_size), 
            loop_vars = [ta, 0]
        )
        ta = ta.stack()
        weights = tf.ones(self.num_patches)
        return ta, weights