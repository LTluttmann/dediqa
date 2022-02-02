# 3rd party
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError, CategoricalCrossentropy, Reduction
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import click
import os
from typing import Union, Tuple

# own modules
from .utils import load_json, list_files, PythonLiteralOption
from .model import DPICNNv2
from .data_generator import DataGenerator, TfDataGenerator
from .callbacks import early_stopper, reducer, SaveCustomModel


class ModelTrainer:
    """class for model training"""
    def __init__(self,
        filepath:str, 
        model_path:str, 
        label_path:str=None, 
        class_labels:list=None, 
        graph_mode:bool=True,
    ):
        self.filepath = filepath
        self.model_path = model_path
        self.class_labels = class_labels 
        self.graph_mode = graph_mode
        self._get_loss()
        if not self.graph_mode:
            print("Graph mode off. Expecting to read jpg files")
            self._get_labels(label_path)
        else:
            print("Graph Mode. Expecting to read files from tfrecords")

    
    def _get_labels(self, label_path):
        # get labels and file paths
        if not label_path:
            self.labels = None
            file_paths = list_files(self.filepath)
            self.file_names = [os.path.split(file_)[-1].split(".")[0] for file_ in file_paths]
        else:
            self.labels = load_json(label_path)
            self.file_names = list(self.labels.keys())

    def _get_loss(self):
        # determine loss function
        if self.class_labels is not None:
            self.loss = CategoricalCrossentropy(reduction=Reduction.SUM)
            self.metric = ['accuracy']
        else:
            self.loss = MeanAbsoluteError(reduction=Reduction.SUM)
            self.metric = None

    def _get_model(self, finetune:bool=False, **model_params):
        """loads existing model if exist and finetuning shall be executed or 
        instantiates a new model otherwise
        args:
        - finetune: specifies whether or not an existing model will be finetuned. If false
        new model is instantiated and learned from scratch
        """
        pretrained_model_path = self.model_path if os.path.exists(self.model_path) and finetune else None
        if pretrained_model_path:
            print(f"load pretrained model from {pretrained_model_path}")
            model = DPICNNv2.load_model(pretrained_model_path)
        else:
            print(f"instantiate new model with basemodel {model_params.get('basemodel')}")
            output_nodes = 1 if self.class_labels is None else len(self.class_labels)
            # instanstiate model
            model = DPICNNv2(
                output_nodes=output_nodes, 
                classes=self.class_labels, 
                **model_params
            )
            model.build(model.base_model.input.shape)
        return model

    def _get_keras_dataset(
        self, 
        model: Model, 
        val_size:float=0.2,
        filter:bool=True,
        **dataset_params
        ):
        """get the keras data generator"""
        # Datasets
        train_files, val_files = train_test_split(self.file_names, test_size=val_size, random_state=123)
        train_samples = [_id for _id in train_files if eval(str(filter).format(name="_id"))]
        val_samples = [_id for _id in val_files if eval(str(filter).format(name="_id"))]   
        # data generators
        training_generator = DataGenerator(
            train_samples,
            img_dir=self.filepath,
            img_crop_dims=model.input_shape_[:2],
            preprocess_input=model.preprocess_input,
            labels=self.labels,
            img_format="jpg",
            output_dim=model.output_nodes,
            classes=self.class_labels,
            **dataset_params
        )

        validation_generator = DataGenerator(
            val_samples,
            img_dir=self.filepath,
            img_crop_dims=model.input_shape_[:2],
            preprocess_input=model.preprocess_input,
            labels=self.labels,
            img_format="jpg",
            output_dim=model.output_nodes,
            shuffle=False,
            random_crops=False,
            classes=self.class_labels,
            **dataset_params
        )

        return training_generator, None, validation_generator, None

    def _get_tf_dataset(
        self, 
        model: Model, 
        batch_size:int,
        crops_per_image:int,
        patch_method:str="random_weighted_patches", 
        filter:str=None,
        **dataset_params
        ) -> Tuple[tf.data.Dataset, Union[int, None], tf.data.Dataset, Union[int, None]]:
        """"""
        batch_size_val = dataset_params.pop("batch_size_val", batch_size)
        patch_method_val = dataset_params.pop("patch_method_val", patch_method)
        crops_per_image_val = dataset_params.pop("crops_per_image_val", crops_per_image)

        # get the tensorflow dataset for training data
        training_generator, train_steps_per_epoch = TfDataGenerator(
            self.filepath.format(kind="train"),
            model.preprocess_input,
            batch_size=batch_size,
            patch_method=patch_method,
            filter=filter,
            num_patches=crops_per_image,
            patch_size=model.input_shape_[0],
            training=True,
            classes=self.class_labels
        ).get_generator(**dataset_params)

        # get the tensorflow dataset for validation data
        validation_generator, val_steps_per_epoch = TfDataGenerator(
            self.filepath.format(kind="val"),
            model.preprocess_input,
            batch_size=batch_size_val,
            patch_method=patch_method_val,
            filter=filter,
            num_patches=crops_per_image_val,
            patch_size=model.input_shape_[0],
            training=False,
            classes=self.class_labels
        ).get_generator(**dataset_params)

        return training_generator, train_steps_per_epoch, validation_generator, val_steps_per_epoch


    def _get_dataset(self, model: Model, **dataset_params):
        """wrapper to get either the tensorflow datagenertor or the keras datagernertor
        args:
        - model: instantiated and fully specified keras Model object 
        """
        if self.graph_mode:
            tg, train_steps, vg, val_steps = self._get_tf_dataset(model, **dataset_params)
        else:
            tg, train_steps, vg, val_steps = self._get_keras_dataset(model, **dataset_params)
        return tg, train_steps, vg, val_steps

    def train(self, params):
        """method to train the DPICNN"""

        # read in the parameters from the settings.json
        params = load_json(params)
        dataset_params = params["dataset_params"]
        fit_params_burnin = params["fit_params_burnin"]
        fit_params_finetune = params["fit_params_finetune"]
        model_params = params["model_params"]

        # get a keras model object (pretrained or not)
        model = self._get_model(**model_params)

        # get datasets. Either tf pipeline or keras sequence pipeline
        training_generator, train_steps, validation_generator, val_steps = self._get_dataset(model, **dataset_params)

        model.base_model.trainable = False
        model.compile(optimizer=Adam(
            learning_rate=fit_params_burnin.pop("learning_rate", 1e-3),
            beta_1=0.9, beta_2=0.999, 
            decay=fit_params_burnin.pop("lr_decay", 1e-3)), 
            loss=self.loss,
            metrics=self.metric
        )

        # start training only dense layers
        model_saver = SaveCustomModel(self.model_path, monitor='val_loss')
        model.summary()
        epochs_burn_in = fit_params_burnin.pop("epochs", 20)
        model.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=epochs_burn_in,
            callbacks=[model_saver, early_stopper, reducer],
            steps_per_epoch=train_steps, 
            validation_steps=val_steps,
            **fit_params_burnin
        )

        # start training all layers
        model.base_model.trainable = True
        model.compile(optimizer=Adam(
            learning_rate=fit_params_finetune.pop("learning_rate", 1e-5),
            beta_1=0.9, beta_2=0.999,  
            decay=fit_params_finetune.pop("lr_decay", 0)), 
            loss=self.loss,
            metrics=self.metric
        )

        model.summary()
        epochs_finetune = fit_params_finetune.pop("epochs", 20)
        model.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=epochs_burn_in + epochs_finetune,
            initial_epoch=epochs_burn_in,
            callbacks=[model_saver, early_stopper, reducer],
            steps_per_epoch=train_steps, 
            validation_steps=val_steps,
            **fit_params_finetune
        )

"""provide cli interface"""
@click.command()
@click.option("-i", "--img_path", type=str, required=True)
@click.option("-l", "--label_path", type=str, default=None)
@click.option("-m", "--model_path", type=str, required=True)
@click.option("-c", "--classes", cls=PythonLiteralOption, default=None)
@click.option("-p", "--params", required=True)
@click.option("--tf/--no-tf", default=True)
def main(img_path, label_path, model_path, classes, params, tf):
    trainer = ModelTrainer(
        filepath=img_path, 
        model_path=model_path, 
        label_path=label_path,
        class_labels=classes,
        graph_mode=tf
    )
    trainer.train(params)
