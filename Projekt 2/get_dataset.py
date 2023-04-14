
from datasets import load_dataset, load_dataset_builder
import tensorflow as tf
import numpy as np
import math as m
from tensorflow.keras import layers
import os

# Remove this folder when making changes in this file
CACHE_PATH = "./tf_cache/"

def _load_from_cache_(split): # I wish it was simpler
    def empty_gen():
        yield 1

    new_ds = tf.data.Dataset.from_generator(empty_gen, output_signature=(
            tf.TensorSpec(shape=(61, 101), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)))

    return _apply_after_cache_decorators_(new_ds, split)

def _apply_after_cache_decorators_(tf_dataset, split):

    tf_dataset = tf_dataset.cache(CACHE_PATH+split)

    if split == "test":
        tf_dataset = tf_dataset.shuffle(10000)

    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset


def get_dataset(split = "test"):

    if os.path.exists(CACHE_PATH+split+".index"):
        return _load_from_cache_(split)

    dataset = load_dataset("speech_commands", "v0.01", split=split)

    def ds_gen():
        for record in dataset:
            x = record["audio"]["array"]
            y = record["label"]
            x_len = x.shape[0]

            # mean = np.mean(x)
            # std = np.std(x)
            # x = (x - mean)/std

            if 10 <= y < 30:
                y = 10 #set others
            if y == 30:
                y = 11 #set silence

            if x_len > 16000:
                xs = np.array_split(x, m.ceil(x_len/16000))
            else:
                xs = [x]

            for a in xs:
                if a.shape[0] < 16000:
                    yield (np.append(a, [0]*(16000 - a.shape[0])), y)
                else:
                    yield (a, y)

    tf_dataset = tf.data.Dataset.from_generator(ds_gen, output_signature=(
        tf.TensorSpec(shape=(16000,), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.int32)))

    def resample(audio):
        return tf.reduce_mean(tf.reshape(audio, (4000, -1)), axis=1)

    # Perhaps resimpling is not necessary wheny we can manipulate frame lenght
    def fourier_transform(audio):
        return tf.abs(tf.signal.stft(audio, frame_length=128, frame_step=64, fft_length=200))

    tf_dataset = tf_dataset.map(
        lambda audio,label: (fourier_transform(resample(audio)), label),
        num_parallel_calls=tf.data.AUTOTUNE)

    # This monster makes caching a toure
    norm = layers.Normalization()
    norm.adapt(data=tf_dataset.map(map_func=lambda x, y: x))

    tf_dataset = tf_dataset.map(
        lambda audio,label: (norm(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)

    if not os.path.exists(CACHE_PATH):
        os.mkdir(CACHE_PATH)

    return _apply_after_cache_decorators_(tf_dataset, split)

def get_categories():
    ds_builder = load_dataset_builder("speech_commands", "v0.01")
    all_categories = dict(enumerate(ds_builder.info.features["label"].names, 0))
    my_categories = {}
    for k, v in all_categories.items():
        if k < 10:
            my_categories[k] = v
            
    my_categories[10] = "others"
    my_categories[11] = "silence"

    return my_categories
