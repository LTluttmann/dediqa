from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from ipywidgets import widgets
import io
from keras.models import load_model
from data_generator import DataGenerator
import os
import json


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

def get_reg_label(label:list):
    filter = np.array(label).astype("bool")
    return LABELS[filter][0]

folder = r"C:\Users\lluttmann\Documents\dev\repos\docubot-2.0\src\backend\dpi-estimation-api\dataset\processed_images"
label_path = r"C:\Users\lluttmann\Documents\dev\repos\docubot-2.0\src\backend\dpi-estimation-api\dataset\labels.json"
path_to_model = r"./cnn_model_fine.h5"
LABELS = np.array([300, 200, 150, 100, 75, 50])

if __name__ == "__main__":
    model = load_model(path_to_model)
    condition = "'no filter'"  # ('ids_' in file or 'passports_' in file)
    files = list_files(folder)
    files = [file for file in files if eval(condition)]
    files = np.random.choice(files, 250, replace=False)
    with open(label_path, "r") as f:
        labels = json.load(f)
    samples = [{"image_id": filename_from_path(path), "label": get_reg_label(labels[filename_from_path(path)])} for path in files]
    data_gen = DataGenerator(samples, folder, 1, preprocess_input, "jpg", shuffle=False)
    y_pred = []
    y_test = []
    for i in range(len(files)):
        preds = []
        for patch in range(20):
            X_test, y = data_gen.__getitem__(i)
            preds.append(model.predict(X_test))
        y_pred.append(np.mean(preds))
        y_test.append(y[0])
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    sort_index = y_test.argsort()
    y_test = y_test[sort_index]
    y_pred = y_pred[sort_index]
    test_array = np.vstack((y_test.ravel(), y_pred.ravel())).T
    mae = np.abs(y_test-y_pred).mean()
    plt.figure(figsize=(20,8))
    plt.plot(test_array[:,0], c="r", label="label")
    plt.bar(list(range(len(test_array[:,1]))), test_array[:,1], label="prediction")
    plt.title(f"Prediction vs. Label: MAE is {round(mae, 2)}")
    plt.legend()
    plt.ylabel("DPI")
    plt.savefig(os.path.join("results", condition))