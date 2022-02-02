from .utils import igetattr, resize_if_too_small
from .data_generator import DataGenerator
from .preprocessing import factory
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
import tensorflow as tf
import cv2
import numpy as np
from scipy.stats import entropy
import importlib
import os
from typing import Union
import pickle
import joblib
from functools import lru_cache as cache


class DPICNNv2(Model):
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

    def __init__(
        self,
        input_shape:tuple,
        basemodel:Union[str, Model],
        dropout_rate:int=0, 
        basemodel_weights:str="imagenet",
        output_nodes:int=1,
        classes:list = None,
        activation:str = "linear",
        mlp_size:int=1024
        ):
        super(DPICNNv2, self).__init__()
        self.input_shape_ = input_shape
        self.output_nodes = output_nodes
        self.dropout_rate = dropout_rate
        self.basemodel = basemodel
        self.basemodel_weights = basemodel_weights
        self.classes = classes
        self.activation = activation
        self.mlp_size = mlp_size
        self._get_base_module()
        self._get_top_model()


    def __getstate__(self):
        weights = self.get_weights()
        model_dict = {
            "input_shape_": self.input_shape_, 
            "output_nodes": self.output_nodes, 
            "dropout_rate": self.dropout_rate,
            "basemodel": self.basemodel,
            "basemodel_weights": self.basemodel_weights,
            "classes": self.classes, 
            "activation": self.activation,
            "preprocess_input": self.preprocess_input,
            "mlp_size": self.mlp_size
        }
        return (model_dict, weights)


    def __setstate__(self, state):
        super(DPICNNv2, self).__init__()
        model_dict, weights = state
        self.__dict__ = {**self.__dict__, **model_dict}
        self._get_base_module()
        self._get_top_model()
        self.build(self.base_model.input.shape)
        self.set_weights(weights)

    def _get_top_model(self):
        self.pooling = GlobalAveragePooling2D()
        self.dense_1 = Dense(self.mlp_size, activation='relu')
        self.dropout_1 = Dropout(self.dropout_rate)
        self.dense_2 = Dense(self.mlp_size, activation='relu')
        self.dropout_2 = Dropout(self.dropout_rate)
        self.regressor = Dense(self.output_nodes , activation=self.activation)

    def _get_base_module(self):
        """
        get base model for doing transfer learning with. If pretrained model for neural image assessment (NIMA) should be
        used, weights for this model can be found here: https://github.com/idealo/image-quality-assessment. It is itself based on
        Mobilenet, hence, mobilenet must be specified as base_model and a path to the NIMA weights must be specified withing 
        basemodel_weight. All other possible base models can be found here: https://keras.io/api/applications/ 
        """
        if isinstance(self.basemodel, str):
            # import module of pretrained basemodel
            base_module = importlib.import_module(f'tensorflow.keras.applications.{self.basemodel.lower()}')
            # get the model definition from the module
            BaseCnn = igetattr(base_module, self.basemodel)
            # get the preprocessing function corresponding to the pretrained model
            self.preprocess_input = getattr(base_module, "preprocess_input")
            if os.path.isfile(self.basemodel_weights):
                # weights can either be a pointer to a file...
                self.base_model = BaseCnn(input_shape=self.input_shape_, include_top=False)
                self.base_model.load_weights(self.basemodel_weights, by_name=True)
            else:
                # ...or the name of the dataset the model is trained on. Most commonly used is 'imagenet'
                self.base_model = BaseCnn(input_shape=self.input_shape_, weights=self.basemodel_weights, include_top=False)
        elif isinstance(self.basemodel, Model):
            # fully specified model was passed
            raise NotImplemented("Not yet implemented. Need to find handy way to determine the preprocessing function corresponding to model --> TODO")
        else:
            raise ValueError("basemodel was not specified correctly, must either be string specifying the name of model or keras model")


    def call(self, inputs):
        x = self.base_model(inputs, training=False)
        x = self.pooling(x)
        x = self.dense_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        return self.regressor(x)

    def predict_img_(self, img, num_patches=None, method="get_convolved_patches_and_weights"):
        """predict one img at a time"""
        if len(img.shape)==4:
            img = img[0,]
        img = resize_if_too_small(img, min_size=self.base_model.input.shape[1]+num_patches)
        patch_generator = getattr(DataGenerator, method)
        patches, weights = patch_generator(img, self.input_shape_[:2], num_patches=num_patches)
        patches = self.preprocess_input(patches)
        preds = self.predict(patches)
        # calc weighted average of patch predictions        
        pred = np.average(preds, axis=0, weights=weights)
        # predict the dot product of class labels and their predicted probabilities
        point_wise = np.array(self.classes).dot(pred)
        # get the class label with highest predicted probability
        class_wise = self.classes[np.argmax(pred)]
        # would probably prefer point_wise 
        return point_wise, class_wise

    def predict_img_tf(self, img, num_patches, method="patches_by_convolution"):
        """predict one img at a time using tensorflow data pipeline"""
        @cache
        def get_patch_generator(num_patches, method):
            """call factory method inside cache decorated func to reduce calls"""
            patch_generator = factory.get_preprocessing(
                name=method, training=False, num_patches=num_patches, 
                patch_size=patch_size, classes=self.classes
            )
            return patch_generator

        def get_otsu(img: tf.Tensor):
            gray = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2GRAY)
            _, otsu = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return np.atleast_3d(otsu)

        patch_size = self.base_model.input.shape[1]
        # get otsu binarization
        otsu = tf.py_function(get_otsu, inp=[img], Tout=tf.uint8)
        # get patches
        patch_generator = get_patch_generator(num_patches, method)
        patches, _, weights = patch_generator(
            img, otsu, [1], *img.shape[:2], resize=None
        )
        patches = self.preprocess_input(tf.cast(patches, tf.float32))
        preds = self.predict(patches)
        pred = np.average(preds, axis=0, weights=weights)
        # predict the dot product of class labels and their predicted probabilities
        point_wise = np.array(self.classes).dot(pred)
        return point_wise

    def save_model(self, path):
        with open(os.path.abspath(path), "wb") as f:
            joblib.dump(self, f)

    @classmethod
    def load_model(cls, f) -> Model:
        if isinstance(f, bytes):
            return joblib.load(f)
        else: 
            with open(f, "rb") as ff:
                model = joblib.load(ff)
            return model


class DPICNN:
    @staticmethod
    def get_weights(otsu_):
        """
        this function determines the weight of a patch for determining the minibatch loss. The weight is simply
        the entropy of the binarized image patch. It is zero (no disorder) if the pixel values of the patch are 
        constant (all pixel values either 0 or 1). The maximum weight is achieved if there are as much black as 
        there are white pixels (assumption: those patches contain the most information)
        """
        p1, p2 = np.bincount(otsu_.ravel(), minlength=2)
        weight = entropy([p1, p2], base=2)
        return weight

    def __init__(
        self, 
        input_shape:tuple,
        basemodel:Union[str, Model],
        learning_rate:int=0.001, 
        dropout_rate:int=0, 
        loss=MeanAbsoluteError(),
        decay:int=0, 
        basemodel_weights:str="imagenet",
        pretrained_model:str=None,
        output_nodes:int=1,
        classes:list = None
        ):
        self.input_shape = input_shape
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.decay = decay
        self.basemodel = basemodel
        self.basemodel_weights = basemodel_weights
        self.pretrained_model = pretrained_model
        self.classes = classes

    def _get_base_module(self):
        """
        get base model for doing transfer learning with. If pretrained model for neural image assessment (NIMA) should be
        used, weights for this model can be found here: https://github.com/idealo/image-quality-assessment. It is itself based on
        Mobilenet, hence, mobilenet must be specified as base_model and a path to the NIMA weights must be specified withing 
        basemodel_weight. All other possible base models can be found here: https://keras.io/api/applications/ 
        """
        if isinstance(self.basemodel, str):
            # import module of pretrained basemodel
            base_module = importlib.import_module(f'tensorflow.keras.applications.{self.basemodel.lower()}')
            # get the model definition from the module
            BaseCnn = igetattr(base_module, self.basemodel)
            # get the preprocessing function corresponding to the pretrained model
            self.preprocess_input = getattr(base_module, "preprocess_input")
            if os.path.isfile(self.basemodel_weights):
                # weights can either be a pointer to a file...
                self.base_model = BaseCnn(input_shape=self.input_shape, include_top=False)
                self.base_model.load_weights(self.basemodel_weights, by_name=True)
            else:
                # ...or the name of the dataset the model is trained on. Most commonly used is 'imagenet'
                self.base_model = BaseCnn(input_shape=self.input_shape, weights=self.basemodel_weights, include_top=False)
        elif isinstance(self.basemodel, Model):
            # fully specified model was passed
            raise NotImplemented("Not yet implemented. Need to find handy way to determine the preprocessing function corresponding to model --> TODO")
        else:
            raise ValueError("basemodel was not specified correctly, must either be string specifying the name of model or keras model")


    def build(self, activation="linear"):
        self._get_base_module()
        inputs = Input(shape=self.input_shape)
        x = self.base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.output_nodes , activation=activation)(x)
        self.model = Model(inputs, outputs)


    def compile(self):
        self.model.compile(optimizer=Adam(lr=self.learning_rate, decay=self.decay), loss=self.loss)


    def predict(self, img):
        """predict one img at a time"""
        if len(img.shape)==4:
            img = img[0,]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if len(img.shape)==3:
            img = np.expand_dims(img, 0)

        # divide img into patches. Use extract patches to get all parts of the image (no randomness involved)
        img_patches = tf.image.extract_patches(
            images=img,
            sizes = (1, *self.input_shape[:2], 1),
            strides= (1, *self.input_shape[:2], 1),
            rates=(1,1,1,1),
            padding="VALID"
        )
        patches = tf.reshape(img_patches,[-1, *self.input_shape])
        # get corresponding otsu patches 
        otsu_patches = tf.image.extract_patches(
            images=np.expand_dims(np.atleast_3d(otsu), 0),
            sizes = (1, *self.input_shape[:2], 1),
            strides= (1, *self.input_shape[:2], 1),
            rates=(1,1,1,1),
            padding="VALID"
        )
        # get weights of patches
        otsu_patches=tf.reshape(otsu_patches,[-1, *self.input_shape[:2]])
        weights = np.array([self.get_weights(otsu_.numpy()) for otsu_ in otsu_patches])
        weights /= weights.sum()  # normalize weights
        # predict the dpi for each patch
        patches = self.preprocess_input(patches.numpy())
        preds = self.model.predict(patches)
        # calc weighted average of patch predictions        
        pred = np.average(preds, axis=0, weights=weights)
        point_wise = np.array(self.classes).dot(pred)
        class_wise = self.classes[np.argmax(pred)]
        # would probably prefer point_wise 
        return point_wise, class_wise

    def __getstate__(self):
        model_dict = self.__dict__.copy()
        model_dict.pop("base_model")
        model_dict.pop("model")
        weights = self.model.get_weights()
        return (model_dict, weights)

    def __setstate__(self, state):
        self.__dict__, weights = state
        self.build()
        self.model.set_weights(weights)

    def save_model(self, path):
        with open(os.path.abspath(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_model(cls, f):
        return joblib.load(f)
