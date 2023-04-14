
from datasets import load_dataset, load_dataset_builder
import tensorflow as tf
import numpy as np
import math as m

def get_dataset(split = "test"):
    dataset = load_dataset("speech_commands", "v0.01", split=split)

    def ds_gen():
        for record in dataset:
            x = record["audio"]["array"]
            y = record["label"]
            x_len = x.shape[0]

            mean = np.mean(x)
            std = np.std(x)
            x = (x - mean)/std

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

    return tf_dataset

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
