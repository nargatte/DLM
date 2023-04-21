from get_dataset import * 
import numpy as np
from tensorflow.keras import models, layers
import os
import json
import copy

BACH_SIZE = 64
REPEAT_COUNT = 15
EPOCHS = 9
BASE_RL = 0.001

ds_train = get_dataset("train").batch(BACH_SIZE)
ds_test = get_dataset("test").batch(BACH_SIZE)

def get_model():
    return models.Sequential([
        layers.Input(shape=[124, 129]),
        layers.Bidirectional(layers.GRU(128)),
        layers.Dense(12)
    ])

tested_lr = [BASE_RL*10**x for x in range(-2, 3)]

def save_json(obj):
    f = open("saved_models/history2.json","w")
    json.dump(obj, f)
    f.close()

full_history = {}

for iter in range(REPEAT_COUNT):
    print(f"REPEAT: {str(iter)}")
    full_history[iter] = {}
    for lr in tested_lr:
        m = get_model()
        m.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        m.fit(
            ds_train,
            validation_data=ds_test,
            epochs=EPOCHS)

        full_history[iter][lr] = copy.deepcopy(m.history.history)
        save_json(full_history)



