
from datasets import load_dataset, load_dataset_builder
import tensorflow as tf
import numpy as np
import math as m
from tensorflow.keras import layers
import os
import random

# Remove this folder when making changes in this file
CACHE_PATH = "./tf_cache/"

def _load_from_cache_(split): # I wish it was simpler
    def empty_gen():
        yield 1

    new_ds = tf.data.Dataset.from_generator(empty_gen, output_signature=(
            tf.TensorSpec(shape=(124, 129), dtype=tf.float32), # Need to change here when shape was changed
            tf.TensorSpec(shape=(), dtype=tf.int32)))

    return _apply_after_cache_decorators_(new_ds, split)

def _apply_after_cache_decorators_(tf_dataset, split):

    tf_dataset = tf_dataset.cache(CACHE_PATH+split)

    if split == "train":
        tf_dataset = tf_dataset.shuffle(10000)

    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset


def get_dataset(split = "test"):

    if os.path.exists(CACHE_PATH+split+".index"):
        return _load_from_cache_(split)

    dataset = load_dataset("speech_commands", "v0.01", split=split)

    def ds_gen_train():
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

            # Stupid subsampling for the biggest of classes 
            # (average_class_size / biggest_class_size)
            if y == 10 and random.random() > (1853 / 32550):
                continue

            if x_len < 16000:
                yield (np.append(x, [0]*(16000 - x_len)), y)
            elif x_len == 16000:
                yield (x, y)
            else: 
                #Slightly less stupid oversampling for the smallest of classes
                part_proportion = x_len / 5411883 # sum of all silence
                part_count = 1853 * part_proportion
                for _ in range(int(part_count)):
                    start = random.randint(0, x_len - 16000 - 1)
                    stop = start + 16000
                    yield (x[start:stop], y)

    def ds_gen_test():
        for record in dataset:
            x = record["audio"]["array"]
            y = record["label"]
            x_len = x.shape[0]

            if 10 <= y < 30:
                y = 10 #set others
            if y == 30:
                y = 11 #set silence

            yield (x, y)

    if split == "test":
        ds_gen = ds_gen_test
    else:
        ds_gen = ds_gen_train

    tf_dataset = tf.data.Dataset.from_generator(ds_gen, output_signature=(
        tf.TensorSpec(shape=(16000,), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.int32)))

    def resample(audio):
        return tf.reduce_mean(tf.reshape(audio, (8000, -1)), axis=1)

    # Perhaps resimpling is not necessary wheny we can manipulate frame lenght
    def fourier_transform(audio):
        return tf.abs(tf.signal.stft(audio, frame_length=255, frame_step=128))

    tf_dataset = tf_dataset.map(
        lambda audio,label: (fourier_transform(audio), label),
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
            
    my_categories[10] = "other"
    my_categories[11] = "silence"

    return my_categories
