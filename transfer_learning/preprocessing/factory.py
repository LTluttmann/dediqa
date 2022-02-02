from . import (
    patches_by_convolution,
    random_non_empty_patches,
    random_weighted_patches
)
from .abstract_class import AbstractPreprocessor

def get_preprocessing(name, training, num_patches, patch_size, classes):
  """Returns preprocessing_fn(image, height, width, **kwargs).
  Args:
  - name: The name of the preprocessing function.
  - training: whether or not the dataset if for training purposes
  - num_patches: number of patches to be croped per image
  - integer specifying size (height and width) of patch
  Returns:
  - preprocessing_fn: A function that preprocesses a single image (pre-batch).
    It has the following signature:
      image = preprocessing_fn(image, otsu, label, height, width, ...)
  Raises:
    ValueError: If Preprocessing `name` is not recognized.
  """
  preprocessing_fn_map = {
      "patches_by_convolution": patches_by_convolution.Preprocessor,
      "random_non_empty_patches": random_non_empty_patches.Preprocessor,
      "random_weighted_patches": random_weighted_patches.Preprocessor
  }

  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)

  def preprocessing_fn(*args, **kwargs):
    preprocessor: AbstractPreprocessor = preprocessing_fn_map[name](training, num_patches, patch_size, classes)
    return preprocessor.preprocess_image(*args, **kwargs)

  return preprocessing_fn