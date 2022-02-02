import numpy as np
import glob
import os
import pandas as pd
import seaborn as sns
from transfer_learning.model import DPICNNv2
from transfer_learning.utils import load_json, ensure_dir_exists, save_json
from transfer_learning.data_generator import TfDataGenerator
# import sys
# sys.path.append(".")
from eval_utils import (
    make_summary_plot, get_reg_label, filename_from_path, make_heatmap
)
project_root = "C:/Users/lluttmann/Documents/dev/repos/docubot-2.0/src/backend/dpi-estimation-api/"
pattern = os.path.join(project_root, "dataset/tfrec_w_tobacco_not_upsc", "dpi_{kind}_*-of-*.tfrecords")


models = glob.glob(os.path.join(project_root, "streamlit_demo", "models", '*.pickle'))

crops_per_image = 14

def get_pred(data, mod):
    i = 0
    y_pred, y_true, dtypes = [], [], []
    for patches, labels, weights, dtype in data:
        if i % 20 == 0:
            print("current iteration: ", i)
        pred = mod.predict(patches)
        mean_pred = np.average(pred, axis=0, weights=weights.numpy())
        point_prediction = mean_pred.dot(np.array(mod.classes))
        y_pred.append(point_prediction)
        # all labels in the batch are the same, get only first one
        reg_label = get_reg_label(labels[0].numpy(), mod.classes)
        y_true.append(reg_label)
        dtypes.append(bytes.decode(dtype.numpy()[0]))
        i += 1
    # create the dataframe
    df = pd.DataFrame({
        "true": y_true, 
        "pred": y_pred,
        "dtype": dtypes
    })
    return df
    

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    # loop over each model in the specified folder
    for model in models:
        # load model
        mod = DPICNNv2.load_model(model)
        # get the generator of test data
        dg, steps = TfDataGenerator(
            tf_records_filepattern=pattern.format(kind="test"), 
            preprocess_input=mod.preprocess_input, 
            batch_size=crops_per_image,  # number of crops per image
            patch_method="random_weighted_patches",
            num_patches=crops_per_image,
            patch_size=mod.input_shape_[0],
            training=False,
            classes=mod.classes
        ).get_generator(resize='random', return_dtype=True, val_set_limit=1000)
        # start prediction process
        pred_df = get_pred(dg, mod)
        make_summary_plot(pred_df, os.path.join(cwd, "results", f"summary_{filename_from_path(model)}_random_upscaled.jpg"))
        make_heatmap(
            pred_df, 
            groupby=("true", "dtype"), 
            label_col="true", pred_col="pred", 
            path=os.path.join(cwd, "results", f"heatmap_{filename_from_path(model)}_random_upscaled.jpg")
        )
