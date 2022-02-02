import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns


def list_files(dir):
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if len(files) > 0:
            for file in files:
                r.append(os.path.join(subdir, file))
    return r


def filename_from_path(path):
    return os.path.split(path)[-1].split(".")[0]


def get_reg_label(label:list, label_list:list):
    filter = np.array(label).astype("bool")
    return np.array(label_list)[filter][0]


def make_preds(model, file, labels, classes=None, num_patches=None):
    img = load_img(file)
    img = np.asarray(img)
    pred1, pred2 = model.predict_img(img, classes, num_patches)
    y = get_reg_label(labels[filename_from_path(file)], model.classes)
    doc_type = filename_from_path(file).split("_")[0]
    return y, pred1, pred2, filename_from_path(file), doc_type


def make_summary_plot(test_df, path=None, label_col="true", pred_col="pred"):
    """function to create a bar like error plot sorted by the different DPIs"""
    # sort according to label
    test_df.sort_values(label_col, inplace=True)
    # calc error metrics
    test_df["error"] = (test_df[label_col]-test_df[pred_col]).abs()
    mae = test_df.error.mean()
    # make the plot
    plt.figure(figsize=(20,8))
    plt.margins(x=0)
    plt.plot(np.arange(0, test_df.shape[0]), test_df[label_col], c="r", label="label")
    plt.bar(list(range(test_df.shape[0])), test_df[pred_col], label="prediction")
    plt.title(f"Prediction vs. Label: MAE is {round(mae, 2)}")
    plt.legend()
    plt.ylabel("DPI")
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()

def make_heatmap(pred_df, groupby:Tuple[str, str], label_col="true", pred_col="pred", path=None):
    # get error heatmap 
    pred_df["error"] = (pred_df[label_col]-pred_df[pred_col]).abs()
    agg = pred_df.groupby([*groupby]).mean()
    agg = agg.reset_index().pivot(index=groupby[0], columns=[groupby[1]], values="error")
    sns.heatmap(agg)
    if path:
        plt.savefig(path)
    else:
        plt.show()
