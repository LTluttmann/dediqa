import click
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from .model import DPICNNv2
import time
"""provide cli interface"""
@click.command()
@click.option("-i", "--image", type=str, required=True)
@click.option("-m", "--model", type=str, required=True)
def main(image, model):
    mod = DPICNNv2.load_model(model)
    img = load_img(image)
    img = np.asarray(img)
    start = time.time()
    for _ in range(1):
        pred = mod.predict_img_tf(img, num_patches=24)
    for _ in range(1):
        pred = mod.predict_img_(img, num_patches=24)
    print("took: ", time.time()-start)
    return pred
