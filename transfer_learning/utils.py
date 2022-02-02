import os
import json
import tensorflow as tf
import numpy as np
import cv2 
import click
import ast

from transfer_learning.preprocessing.utils import min_resize


class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(str(value))
        except:
            raise click.BadParameter(value)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, target_file):
    with open(target_file, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_image(img_file):
    return np.asarray(tf.keras.preprocessing.image.load_img(img_file), dtype="uint8")


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def resize_if_too_small(img, min_size:int=224):
    h, w = img.shape[:2]
    scale = max(1, max(min_size/h, min_size/w))
    new_size = (int(np.ceil(w*scale)), int(np.ceil(h*scale)))
    img = cv2.resize(img, new_size)
    return img


def list_files(dir):
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if len(files) > 0:
            for file in files:
                r.append(os.path.join(subdir, file))
    return r

def change_res_quality(img, resize_frac):
    img = img.copy()
    img = cv2.GaussianBlur(img, (3,3), 0)
    im_res = cv2.resize(img, (int(resize_frac * img.shape[1]), int(resize_frac * img.shape[0])))
    im_res = cv2.resize(im_res, (img.shape[1], img.shape[0]))
    return im_res

def change_res(img, resize_frac):
    img = img.copy()
    im_res = cv2.resize(img, (int(resize_frac * img.shape[1]), int(resize_frac * img.shape[0])))
    im_res = cv2.resize(im_res, (img.shape[1], img.shape[0]))
    return im_res

def quality_downsample_tf(img, resize_frac):
    img = img.copy()
    im_res = tf.image.resize(img, size=(np.array(img.shape[:2])*resize_frac).astype("int"), antialias=True)
    return im_res.numpy().astype(np.uint8)


def downsample_tf(img, resize_frac):
    img = img.copy()
    im_res = cv2.resize(img, (int(resize_frac * img.shape[1]), int(resize_frac * img.shape[0])))
    return im_res

def igetattr(obj, attr):
    """case insensitive getattr implementation. Helpful for import pretrained Keras models."""
    for a in dir(obj):
        if a.lower() == attr.lower():
            return getattr(obj, a)