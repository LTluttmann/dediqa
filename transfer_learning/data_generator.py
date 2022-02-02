import functools
from .utils import load_image, change_res_quality, change_res
from .preprocessing import factory
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.stats import entropy
from typing import Callable, Tuple, Union
from functools import reduce


class TfDataGenerator(object):
    """
    data generator implemented with the tf.data api. This implementation
    is much faster than the keras data generator
    """
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'otsu': tf.io.FixedLenFeature([], tf.string),
        'doc_type': tf.io.FixedLenFeature([], tf.string),
    }

    def __init__(
        self, 
        tf_records_filepattern:str, 
        preprocess_input:Callable,
        batch_size:int,
        num_patches:int,
        patch_size:int,
        filter:str=None,
        patch_method:str="random_weighted_patches",
        training:bool=False,
        classes:list=None
        ):
        self.file_pattern = tf_records_filepattern
        self.preprocess_input = preprocess_input
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.training = training
        self.patch_method = factory.get_preprocessing(
            patch_method, training, num_patches, patch_size, classes)
        self._get_filter_condition(filter)

    def _get_filter_condition(self, filter_cond:None):
        """
        function to define a filter funtion given a filter condition. Currently, the only 
        valid filter conditions are 'ids', where only ids remain in the dataset or 'no ids'
        where all ids are discarded from the dataset
        args:
            - filter_cond: either None, 'ids' or 'no ids'
        """
        if filter_cond is None:
            self.filter = lambda *args: tf.constant(True)
        elif filter_cond.lower() == "ids":
            self.filter = lambda *args: tf.math.equal(args[-1], tf.constant(str.encode("ids")))
        elif filter_cond.lower() == "no ids":
            self.filter = lambda *args: tf.math.not_equal(args[-1], tf.constant(str.encode("ids")))
        else:
            raise ValueError("filter condition must be any of None, 'IDs' or 'no IDs'")

    def _parse_image_function(self, example_proto):
        """function to parse image and metadata from tfrecords
        Tfrecords are expected to contain the image, its otsu binarization,
        height, width and depth of the image as well as the label and document type
        This function parses this information and returns it 
        """
        # Parse the input tf.train.Example proto using the dictionary above
        features = tf.io.parse_single_example(example_proto, self.image_feature_description)
        # get image shape for reconstructing otsu binarization from bytes
        height = tf.cast(features["height"], tf.int64)
        width = tf.cast(features["width"], tf.int64)
        depth = tf.cast(features["depth"], tf.int64)
        # get the original image from bytes
        image = tf.io.decode_jpeg(features["image"], channels=3)
        # get otsu binarization of image
        otsu = tf.io.decode_jpeg(features["otsu"], channels=1)
        otsu = tf.reshape(otsu, (height, width, 1))
        # get label from bytes
        label = tf.io.parse_tensor(features["label"], out_type=tf.int32)
        return image, otsu, label, height, width, features["doc_type"]

    def _get_training_dataset(
        self, buffer_size:int=10000, **kwargs
        ) -> Tuple[tf.data.Dataset, Union[int,None]]:
        """"""
        # list files matching the pattern
        files = tf.data.Dataset.list_files(self.file_pattern, shuffle=True)
        num_files = tf.data.Dataset.cardinality(files)
        # steps_per_epoch = int(np.floor((num_files * 200) / self.batch_size))
        filtered_dataset = (
            files
            .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE, cycle_length=num_files, deterministic=False)
            .map(self._parse_image_function, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            .filter(self.filter)
        )
        num_files = filtered_dataset.reduce(0, lambda x,_: x+1).numpy()
        steps_per_epoch = int(np.floor((num_files) / self.batch_size))
        dataset = (
            filtered_dataset.map(
                lambda img, otsu, label, height, width, *_: 
                self.patch_method(img, otsu, label, height, width, **kwargs), 
                num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
            )
            .unbatch()  # stacking tensors of patches together
            .map(
                lambda x, y, w: (self.preprocess_input(tf.cast(x, tf.float32)), y, w), 
                num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
            )
            .shuffle(buffer_size)  # buffer_size=self.batch_size * self.num_patches)  
            .batch(self.batch_size)
            .map(
                lambda x, y, w: [x, y, tf.math.divide(w, tf.math.reduce_sum(w))], 
                num_parallel_calls=tf.data.AUTOTUNE
            )  # normalize weights for batch
            .repeat()
            .prefetch(tf.data.AUTOTUNE) 
        )
        return dataset, steps_per_epoch

    def _get_test_dataset(
        self, val_set_limit:int=-1, return_dtype:bool=False, **kwargs
        ) -> Tuple[tf.data.Dataset, Union[int, None]]:
        """"""
        files = tf.data.Dataset.list_files(self.file_pattern, shuffle=False)
        num_files = tf.data.Dataset.cardinality(files)
        dataset = (
            files
            .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE, cycle_length=num_files)
            .map(self._parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
            .filter(self.filter)
            .map(
                lambda img, otsu, label, height, width, dtype: 
                self.patch_method(img, otsu, label, height, width, **kwargs) if not return_dtype
                else [*self.patch_method(img, otsu, label, height, width, **kwargs), 
                tf.tile(tf.expand_dims(dtype, 0), tf.constant([self.num_patches,]))],
                num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
            )
            .unbatch()  # stacking tensors of patches together
            .map(
                lambda x, y, w, *_: (self.preprocess_input(tf.cast(x, tf.float32)), y, w, *_), 
                num_parallel_calls=tf.data.AUTOTUNE
            ) 
            .batch(self.batch_size)
            .take(val_set_limit)  # limit validation set
            .map(
                lambda x, y, w, *_: [x, y, tf.math.divide(w, tf.math.reduce_sum(w)), *_], 
                num_parallel_calls=tf.data.AUTOTUNE)  # normalize weights for batch
            .prefetch(tf.data.AUTOTUNE) 
        )
        return dataset, None

    def get_generator(self, **kwargs) -> Tuple[tf.data.Dataset, Union[int,None]]:
        if self.training:
            return self._get_training_dataset(**kwargs)
        else:
            return self._get_test_dataset(**kwargs)


# old dataclass. Should always use tf dataclass, since it is faster
class DataGenerator(tf.keras.utils.Sequence):
    """custom data generator class to process images 'on the go' as required"""

    @staticmethod
    def get_sample_weights(otsu_):
        """
        this function determines the weight of a patch for determining the minibatch loss. The weight is simply
        the entropy of the binarized image patch. It is zero (no disorder) if the pixel values of the patch are 
        constant (all pixel values either 0 or 1). The maximum weight is achieved if there are as much black as 
        there are white pixels (assumption: those patches contain the most information)
        """
        p1, p2 = np.bincount(otsu_.ravel(), minlength=2)
        weight = entropy([p1, p2], base=2)
        return weight

    @staticmethod
    def get_random_patches_and_weights(img, patch_dim):
        """
        crop random patch and derive its weight in terms of model loss. Patches which only consist of white
        background should have zero weight, as the model cannot derive meaningful features from it. This is achieved using
        binary entropy score of the corresponding patch from the otsu binarized image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = img.shape[0], img.shape[1]
        ch, cw = patch_dim
        offset_w = np.random.randint(0, w - cw + 1)
        offset_h = np.random.randint(0, h - ch + 1)
        patch = img[offset_h:(offset_h+ch), offset_w:(offset_w+cw), :]
        weight = DataGenerator.get_sample_weights(otsu[offset_h:offset_h+h, offset_w:offset_w+w].ravel())
        return patch, weight

    @staticmethod
    def _determine_stride_size(img_shape, num_patches, patch_size):
        h, w = img_shape[0], img_shape[1]
        factors = reduce(list.__add__, ([i, num_patches//i] for i in range(1,int(num_patches**0.5)+1) if num_patches % i == 0))
        
        aspect_ratio = min([h,w]) / max([h,w])
        fracs = {(factors[i],factors[i+1]): np.abs((factors[i]/factors[i+1])-aspect_ratio) for i in range(0, len(factors), 2)}
        
        patches = min(fracs, key=fracs.get)
        num_patches_h, num_patches_w = (min(patches), max(patches)) if h<w else (max(patches), min(patches))
        num_patches_h, num_patches_w = num_patches_h-1, num_patches_w-1
        stride_h = np.floor((h-patch_size[0]) / num_patches_h).astype("int")
        stride_w = np.floor((w-patch_size[1]) / num_patches_w).astype("int")
        return stride_h, stride_w


    @staticmethod
    def get_convolved_patches_and_weights(img, patch_dim, norm_weights=True, num_patches=None):
        """
        crop patches through convolution and derive their weights in terms of model loss. Patches which only consist of white
        background should have zero weight, as the model cannot derive meaningful features from it. This is achieved using
        binary entropy score of the corresponding patch from the otsu binarized image
        """
        stride_size = patch_dim if num_patches is None else DataGenerator._determine_stride_size(img.shape, num_patches, patch_dim)
        channels = np.atleast_3d(img).shape[-1]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined = tf.concat((img, np.atleast_3d(otsu)), axis=-1)
        # divide img into patches. Use extract patches to get all parts of the image (no randomness involved)
        patches = tf.image.extract_patches(
            images=np.expand_dims(combined, 0),
            sizes = (1, *patch_dim, 1),
            strides= (1, *stride_size, 1),
            rates=(1,1,1,1),
            padding="VALID"
        )
        patches = tf.reshape(patches,[-1, *patch_dim, channels+1])
        img_patches = tf.cast(patches[:,:,:,:3], tf.float32)
        otsu_patches = patches[:,:,:,-1]
        weights = np.array([DataGenerator.get_sample_weights(otsu_.numpy()) for otsu_ in otsu_patches])
        weights = weights / weights.sum() if norm_weights else weights  # normalize weights if demanded
        # predict the dpi for each patch
        return img_patches, weights        


    def __init__(
        self, 
        samples:list, 
        img_dir:str, 
        preprocess_input:Callable,
        labels: dict=None,
        batch_size:int=64, 
        img_format:str="jpg", 
        img_crop_dims:tuple=(224, 224),  # don't change when using mobilenet/NIMA
        channels:int=3,  # don't change when using mobilenet/NIMA
        output_dim:int=1,
        shuffle:bool=True,
        flow:bool=True,
        data_augmentation:ImageDataGenerator=None,
        classes:list=None,
        n_crops_per_image:int=1,
        random_crops:bool=True
    ):
        """
        :param samples: list of file names
        :param img_dir: path to images
        :param batch_size: batch size
        :param img_format: format the image is stored on disk
        :param img_crop_dims: the size of the random patches to be generated during training. Must match the input size of network
        :param channels: channels of image (3 for rgb). Must match the network input
        :param output_dim: number of output nodes of the model (1 for regression, number of classes for classification)
        :param shuffle: whether or not to shuffle the training data after each epoch. Turn of during validation/prediciton phase!
        :param flow: specifies whether to read images from disk every epoch for each image (slow) or to store images in memory once and read from there (requires a lot of memory)
        :param preprocessing_func: function specifying any additional preprocessing functionalities
        """
        self.samples = samples
        self.img_dir = img_dir
        self.preprocess_input = preprocess_input
        self.labels = labels
        self.batch_size = batch_size
        self.img_format = img_format
        self.img_crop_dims = img_crop_dims  # dimensions that images get randomly cropped to
        self.channels = channels
        self.output_dim = output_dim
        self.shuffle = shuffle
        self.flow = flow
        self.data_augmentation = data_augmentation
        self.classes = classes
        self.n_crops_per_image = n_crops_per_image
        self.random_crops = random_crops
        self.on_epoch_end()  # call ensures that samples are shuffled in first epoch if shuffle is set to True
        if not self.flow:  # load all data into ram
            self._load_and_store_images()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        if self.labels is not None:
            X, y, w = self.__data_generator(batch_samples)
        else:
            X, y, w = self.__data_and_label_generator(batch_samples)
        return X, y, w

    def _load_and_store_images(self):
        self.img_dict = {}
        for train_img in self.samples:
            img_file = os.path.join(self.img_dir, '{}.{}'.format(train_img, self.img_format))
            img = load_image(img_file)
            self.img_dict[train_img] = img

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generator(self, batch_samples):
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples) * self.n_crops_per_image, *self.img_crop_dims, self.channels))
        w = np.empty((len(batch_samples) * self.n_crops_per_image), dtype=float)
        y = np.empty((len(batch_samples) * self.n_crops_per_image, self.output_dim))

        count = 0
        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            if not self.flow:
                img = self.img_dict[sample]
            else:
                img_file = os.path.join(self.img_dir, '{}.{}'.format(sample, self.img_format))
                img = load_image(img_file)
            if self.random_crops:
                for _ in range(self.n_crops_per_image): 
                    # img = resize_if_too_small(img, (256, 256))
                    patch, weight = self.get_random_patches_and_weights(img, self.img_crop_dims)
                    X[count, ] = patch
                    w[count] = weight

                    # normalize labels
                    y[count, ] = self.labels[sample]
                    count += 1
            else:
                idx1 = i * self.n_crops_per_image
                idx2 = idx1 + self.n_crops_per_image
                patches, weights = self.get_convolved_patches_and_weights(
                    img, self.img_crop_dims, norm_weights=False, num_patches=self.n_crops_per_image
                    )
                X[idx1:idx2, ] = patches
                w[idx1:idx2, ] = weights
                y[idx1:idx2, ] = self.labels[sample]

        w /= w.sum()
        # apply basemodel specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.preprocess_input(X)
        if self.data_augmentation:
            X_gen = self.data_augmentation.flow(X, batch_size=X.shape[0], shuffle=False)
            X = next(X_gen)
        return X, y, w

    def _get_random_dpi_img(self, img):
        """function randomly downsamples image (or maintains the original one)"""
        img = img.copy()
        new_dpi = np.random.choice(self.classes)
        resize_frac = new_dpi / float(max(self.classes))  # we assume the original doc has a dpi of 300
        if resize_frac < 1:
            # Downsample img. Use quality downsampling 50% of the time. The other 50% use plain down- and upsampling
            if np.random.random() < .5:
                img = change_res_quality(img, resize_frac)
            else:
                img = change_res(img, resize_frac)
            return img, new_dpi
        else:
            # return original image and dpi of 300
            return img, new_dpi

    def __data_and_label_generator(self, batch_samples):
        assert self.classes, "classes have to be specified if no labels are provided"
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples) * self.n_crops_per_image, *self.img_crop_dims, self.channels))
        w = np.empty((len(batch_samples) * self.n_crops_per_image), dtype=float)
        y = np.empty((len(batch_samples) * self.n_crops_per_image, self.output_dim))

        count = 0
        for sample in batch_samples:
            # load and randomly augment image
            if not self.flow:
                img = self.img_dict[sample]
            else:
                img_file = os.path.join(self.img_dir, '{}.{}'.format(sample, self.img_format))
                img = load_image(img_file)
            
            for _ in range(self.n_crops_per_image): 
                new_img, label = self._get_random_dpi_img(img)
                patch, weight = self.get_random_patches_and_weights(new_img, self.img_crop_dims)
                X[count, ] = patch
                w[count] = weight

                # normalize labels
                y[count, ] = (np.array(self.classes) == label).astype("int")
                count += 1

        w /= w.sum()
        # apply mobilenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.preprocess_input(X)
        if self.data_augmentation:
            X_gen = self.data_augmentation.flow(X, batch_size=X.shape[0], shuffle=False)
            X = next(X_gen)
        return X, y, w
