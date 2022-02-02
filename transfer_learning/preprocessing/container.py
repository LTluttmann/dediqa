# import tensorflow as tf

# def _get_patches_by_convolution(self, combined, h, w):
#     """
#     for a given number of patches to be extracted, this method first finds the stride
#     size so that a convolutional patch extraction yields the desired number of patches.

#     For convolutional filters, the output size is determined by:
#     num_filter = ((input_size-filter_size)/stride)+1

#     Hence, the stride size for given number of filters (patches) is as follows:
#     s = (input_size - filter_size) / (num_filters - 1)

#     The determined stride size is then used in tf.image.extract_patches to crop the patches.
#     This function must be executed in eager mode, since the determined stride size may not 
#     be a tensor but an integer

#     args:
#     - combined: the concatentation of the input image and its otsu binarization
#     - h: height of the input image
#     - w: width of the input image
#     """
#     # determine the factors of num_patches 
#     factors = functools.reduce(list.__add__, ([(i, self.num_patches//i)] for i in range(
#         1, int(np.sqrt(self.num_patches))+1) if self.num_patches % i == 0))
#     # determine the aspect ratio of the image
#     aspect_ratio = tf.math.divide(tf.math.minimum(h, w), tf.math.maximum(h, w))
#     # choose the factors, which ratio is closest to the aspect ratio of the image
#     # For example, if 4 images of a square image have to be extracted, we want to
#     # extract 2 images per "column" instead of all patches lying in a line
#     idx = tf.argmin([tf.abs(tf.subtract(tf.divide(*factors[i]), aspect_ratio)) for i in range(0, len(factors), 1)])
#     patch1, patch2 = tf.unstack(tf.gather(tf.constant(factors), idx))
#     # extract more patches from the larger side of the image
#     if tf.less(h,w):
#         num_patches_h, num_patches_w = tf.minimum(patch1, patch2), tf.maximum(patch1, patch2)
#     else:
#         num_patches_h, num_patches_w = tf.maximum(patch1, patch2), tf.minimum(patch1, patch2)
#     # determine stride size according to formula given in description
#     num_patches_h, num_patches_w = tf.subtract(num_patches_h, 1), tf.subtract(num_patches_w, 1)
#     stride_h = tf.cast(tf.math.floor(tf.divide(tf.cast(tf.subtract(h, self.patch_size), tf.int64), tf.cast(num_patches_h, tf.int64))), tf.int16)
#     stride_w = tf.cast(tf.math.floor(tf.divide(tf.cast(tf.subtract(w, self.patch_size), tf.int64), tf.cast(num_patches_w, tf.int64))), tf.int16)
    
#     # extract and return patches
#     patches = tf.reshape(
#         tf.image.extract_patches(
#             images=tf.expand_dims(combined, 0),
#             sizes=[1, self.patch_size, self.patch_size, 1],
#             strides=[1, stride_h.numpy(), stride_w.numpy(), 1],
#             rates=[1, 1, 1, 1],
#             padding='VALID'), 
#         (-1, self.patch_size, self.patch_size, 4))

#     return patches

# # @tf.function
# def get_random_patches_and_weights(self, image, otsu, label, *args):
#     """Get `num_patches` random crops from the image"""
#     # init lists containing the patches
#     patches, weights = [], []
#     # make list with label
#     labels = tf.tile(tf.expand_dims(label, 0), tf.constant([self.num_patches, 1]))
#     # concat original image and otsu to ensure same patches are cropped
#     combined = tf.concat([image, otsu], axis=-1)
#     for _ in range(self.num_patches):
#         patch = tf.image.random_crop(combined, [self.patch_size, self.patch_size, 4])
#         patches.append(tf.cast(patch[:,:,:3], tf.float32))
#         otsu_patch = patch[:,:,-1]
#         bins = tf.math.bincount(tf.cast(otsu_patch, tf.int32), minlength=2, maxlength=2)
#         entropy = self.tf_entropy(bins)
#         weights.append(entropy)
#     patches = tf.stack(patches)
#     weights = tf.stack(weights)
#     assert patches.get_shape().dims == [self.num_patches, self.patch_size, self.patch_size, 3]

#     return patches, labels, weights


# def get_patches(self, img, otsu, label, height, width, *args):
#     """crop deterministic patches by convolution"""
#     combined = tf.concat([img, otsu], axis=-1)
#     max_dpi = 300
#     label = tf.reshape(label, (6,))
#     scale =tf.cast(tf.reduce_sum(tf.math.divide(max_dpi, tf.boolean_mask(
#         tf.constant([300,210,150,100,75,50]), label))), dtype=tf.float32)
#     height = tf.cast(tf.multiply(tf.cast(height, tf.float32), scale), tf.int32)
#     width = tf.cast(tf.multiply(tf.cast(width, tf.float32), scale), tf.int32)
#     combined = tf.image.resize(combined, size=[height, width])

#     patches = tf.py_function(
#         self._get_patches_by_convolution, 
#         inp=[combined, height, width], 
#         Tout=tf.float32)
#     img_patches = tf.cast(patches[:,:,:,:3], tf.float32)
#     otsu = tf.cast(patches[:,:,:,-1], tf.int32)
#     weights = tf.map_fn(lambda x: self.tf_entropy(x), otsu, fn_output_signature=tf.float64)
#     labels = tf.tile(tf.expand_dims(label, 0), tf.constant([self.num_patches, 1]))
#     return img_patches, labels, weights


# def random_resize_and_crop(self, img, otsu, label, height, width, *args):
#     """function to be used for scale jittering"""
#     max_dpi = 300
#     height = tf.cast(height, dtype=tf.float32)
#     width = tf.cast(width, dtype=tf.float32)

#     label_scalar = tf.boolean_mask(tf.constant([300,210,150,100,75,50]), tf.reshape(label, (6,)))
#     scale_upper = tf.cast(tf.reduce_sum(tf.math.divide(max_dpi, label_scalar)), dtype=tf.float32)
#     scale_lower = tf.maximum(1.0, tf.maximum(tf.divide(self.patch_size, height), tf.divide(self.patch_size, width)))

#     scale_jittered_patches, weights = [], []
#     labels = tf.tile(tf.expand_dims(label, 0), tf.constant([self.num_patches, 1]))
#     combined = tf.concat([img, otsu], axis=-1)

#     scales = tf.random.uniform([self.num_patches], minval=scale_lower, maxval=scale_upper, dtype=tf.float32)
#     for i in range(self.num_patches):
#         # scale = tf.random.uniform([], minval=scale_lower, maxval=scale_upper, dtype=tf.float32, seed=None, name=None)
#         scale = tf.gather(scales, i)
#         resized = tf.image.resize(combined, tf.cast([tf.multiply(height, scale), tf.multiply(width, scale)], tf.int32))
#         patch = tf.image.random_crop(resized, [self.patch_size, self.patch_size, 4])
        
#         img_patch = tf.cast(patch[:,:,:3], tf.float32)
#         scale_jittered_patches.append(img_patch)
        
#         otsu_patch = tf.cast(patch[:,:,-1], tf.uint8)
#         # w = tf.cond(tf.reduce_all(tf.math.equal(otsu_patch, tf.cast(tf.ones_like(otsu_patch), dtype=tf.float32))), lambda: 0, lambda: 1)
#         bins = tf.math.bincount(tf.cast(otsu_patch, tf.int32), minlength=2, maxlength=2)
#         entropy = self.tf_entropy(bins)
        
#         weights.append(entropy)        
        
#     return tf.stack(scale_jittered_patches), labels, tf.stack(weights)


# def random_resize_and_crop_single(self, img, otsu, label, height, width, *args):
#     height = tf.cast(height, dtype=tf.float32)
#     width = tf.cast(width, dtype=tf.float32)

#     classes = tf.tile(tf.expand_dims(tf.constant([300,210,150,100,75,50]), 0), tf.constant([self.batch_size, 1]))
#     label = tf.reshape(label, tf.shape(classes))
#     label_scalar = tf.boolean_mask(classes, label)
#     scale_upper = tf.cast(tf.reduce_sum(tf.math.divide(300, label_scalar)), dtype=tf.float32)
#     scale_lower = tf.cast(tf.maximum(1.0, tf.maximum(tf.divide(self.patch_size, height), tf.divide(self.patch_size, width))), dtype=tf.float32)
    
#     combined = tf.concat([img, otsu], axis=-1)

#     scale = tf.random.uniform([self.batch_size], minval=scale_lower, maxval=scale_upper, dtype=tf.float32, seed=None, name=None)
#     resized = tf.image.resize(combined, tf.cast([height*scale, width*scale], tf.int32))
#     patch = tf.image.random_crop(resized, [self.patch_size, self.patch_size, 4])

#     img_patch = patch[:,:,:,:3]
#     otsu_patch = tf.cast(patch[:,:,:,-1], tf.uint8)     
#     w = tf.cond(tf.reduce_all(tf.math.equal(otsu_patch, tf.cast(tf.ones_like(otsu_patch), dtype=tf.uint8))), lambda: 0, lambda: 1)    
        
#     return img_patch, label, w

# def random_crop_single(self, img, otsu, label, height, width, *args):
#     combined = tf.concat([img, otsu], axis=-1)
#     patch = tf.image.random_crop(combined, [self.patch_size, self.patch_size, 4])

#     img_patch = tf.cast(patch[:,:,:3], tf.float32)
#     otsu_patch = tf.cast(patch[:,:,-1], tf.int32)      
#     bins = tf.math.bincount(otsu_patch, minlength=2, maxlength=2)
#     w = self.tf_entropy(bins)    
#     return img_patch, label, w