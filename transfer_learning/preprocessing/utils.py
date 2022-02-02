import tensorflow as tf


MAX_DPI = 300


def tf_entropy(bins: tf.Tensor):
    """
    this function calculated the entropy:
    e = -sum(p(i) * log(p(i)) over all i)

    The entropy is used as a weight for a patch, since the entropy exhibits 
    higher values on a otsu binarized patch if more "content" is visible in 
    the batch and zero is there is no content (constant pixel values)

    args:
    - bins: bin counts for the distinct pixel values. Pixel values are either 
    0 or 1, since we use the otsu binarized images for entropy determination
    """
    p = tf.math.divide(bins, tf.math.reduce_sum(bins))
    has_nans = tf.negative(tf.math.multiply(p, tf.math.log(p)))
    has_no_nans = tf.where(tf.math.is_nan(has_nans), tf.zeros_like(has_nans), has_nans) 
    return tf.math.reduce_sum(has_no_nans)


def get_weight(otsu_patch):
    """Determines the weight of an image patch given its otsu binarization. 
    args:
    - otsu binarizatin of the patch to be evaluated
    """
    otsu_patch = tf.cast(otsu_patch, tf.int32)
    bins = tf.math.bincount(otsu_patch, minlength=2, maxlength=2)
    entropy = tf_entropy(bins)
    return entropy


def random_resize(img, height, width, label, patch_size=224, classes=None, **_):
    """resizes images using a random scalar. The scalar is is drawn uniformly from the range
    of a minimum scalar factor, which is needed so that patches of the specified size can be 
    cropped from the image, and a maximum scalar determined by the fraction of the maximum possible 
    dpi and the dpi of the Image to be resized.
    Args:
    - img: Image to be resized
    - patch_size: integer specifying hight and width of a random crop
    - heigh, width: specify the current height and width of the image
    Returns:
    - resized: resized image
    """
    print("perform random resizing")
    height = tf.cast(height, dtype=tf.float32)
    width = tf.cast(width, dtype=tf.float32)

    if classes is not None:
        max_dpi = max(classes)
        label_scalar = tf.boolean_mask(tf.constant(classes), tf.reshape(label, (len(classes),)))
    else:
        max_dpi = MAX_DPI
        label_scalar = label

    scale_upper = tf.cast(tf.reduce_sum(tf.math.divide(max_dpi, label_scalar)), dtype=tf.float32)
    scale_lower = tf.maximum(1.0, tf.maximum(tf.divide(patch_size, height), tf.divide(patch_size, width)))

    scale = tf.random.uniform([], minval=scale_lower, maxval=scale_upper, dtype=tf.float32)
    if scale <= 1.1 and scale_lower==1.0:
        # we also want to have unscaled images in the training set
        return tf.cast(img, tf.float32)
    else:
        return tf.image.resize(img, tf.cast([height*scale, width*scale], tf.int32))


def min_resize(img, height, width, *__, patch_size=224, num_patches=0, **_):
    """resizes the image to the minimum necesarry size so that patches of the
    specified size can be cropped from it. If one side of the image is smaller than
    the patch size, the fraction of the patch_size and the pixels of the image determine
    the scale, by which the image has to be upscaled. The maximum scale of both sides has
    to be used so that patches can be cropped.
    Args:
    - img: Image to be resized
    - patch_size: integer specifying hight and width of a random crop
    - heigh, width: specify the current height and width of the image
    Returns:
    - resized: resized image
    """
    print("perform min resizing")
    height = tf.cast(height, dtype=tf.float32)
    width = tf.cast(width, dtype=tf.float32)
    scale = tf.maximum(1.0, tf.maximum(tf.divide(patch_size, height), tf.divide(patch_size, width)))
    if scale==1.0:
        return tf.cast(img, tf.float32)
    else:
        new_height = tf.cast(tf.math.ceil(height*scale), tf.int32) + num_patches
        new_width = tf.cast(tf.math.ceil(width*scale), tf.int32) + num_patches
        resized = tf.image.resize(img, (new_height, new_width))
        return resized


def max_resize(img, height, width, label, *__, classes=None, **_):
    """resizes the image to the size of the maximum DPI. If the image has a DPI of 75 
    and the maximum DPI of the training set is 300, the image is upscaled by a factor of 4.
    This leads to images of the same classes having roughly the same size
    Args:
    - img: Image to be resized
    - label: label specifying dpi of image
    - heigh, width: specify the current height and width of the image
    - classes: possible dpis of training set
    Returns:
    - resized: resized image
    """
    print("perform max resizing")
    height = tf.cast(height, dtype=tf.float32)
    width = tf.cast(width, dtype=tf.float32)
    if classes is not None:
        max_dpi = max(classes)
        label_scalar = tf.boolean_mask(tf.constant(classes), tf.reshape(label, (len(classes),)))
    else:
        max_dpi = MAX_DPI
        label_scalar = label
    scale = tf.cast(tf.reduce_sum(tf.math.divide(max_dpi, label_scalar)), dtype=tf.float32)
    resized = tf.image.resize(img, tf.cast([height*scale, width*scale], tf.int32))
    return resized