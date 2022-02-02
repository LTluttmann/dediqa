from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import numpy as np


class SaveCustomModel(Callback):
    def __init__(self, model_path:str, monitor:str="val_loss"):
        # self.custom_model = model
        self.model_path = model_path
        self.monitor = monitor
        self.monitor_op = np.less
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                    ' saving model to %s' % (epoch + 1, self.monitor,
                                            self.best, current, self.model_path))
            self.best = current
            self.model.save_model(self.model_path)
        else:
            print('\nEpoch %05d: %s did not improve from %0.5f' %
                    (epoch + 1, self.monitor, self.best))

class CheckWeights(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            "The weights of first conv layers feature map are ",
            self.model.layers[1].get_weights()[0][0][0][0]
        ) 

weight_checker = CheckWeights()


model_checkpointer = ModelCheckpoint(
    filepath="./model.h5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)

early_stopper = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)

reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1)