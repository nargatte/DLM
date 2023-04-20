from get_dataset import * 
import numpy as np
from tensorflow.keras import models, layers
import os
import json
import copy

BACH_SIZE = 64
REPEAT_COUNT = 15
MODELS_FOLDER = "saved_models"
MAX_EPOCHS = 30

if not os.path.exists(MODELS_FOLDER):
    os.mkdir(MODELS_FOLDER)

ds_train = get_dataset("train")
ds_test = get_dataset("test")

def my_map(x, y):
    return x[..., tf.newaxis], y

ds_train_conv = ds_train.map(my_map)
ds_test_conv = ds_test.map(my_map)

def get_models():
    conv_model = models.Sequential([
        layers.Input(shape=[124, 129, 1]),
        layers.Resizing(32, 64),
        layers.Conv2D(32, 3, activation='gelu'),
        layers.Conv2D(32, 4, activation='gelu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='gelu'),
        layers.Dropout(0.5),
        layers.Dense(12),
    ])

    rnn_model = models.Sequential([
        layers.Input(shape=[124, 129]),
        layers.Bidirectional(layers.SimpleRNN(64)),
        layers.Dense(12)
    ])

    lstm_model = models.Sequential([
        layers.Input(shape=[124, 129]),
        layers.Bidirectional(layers.LSTM(128)),
        layers.Dense(12)
    ])

    gru_model = models.Sequential([
        layers.Input(shape=[124, 129]),
        layers.Bidirectional(layers.GRU(128)),
        layers.Dense(12)
    ])

    models_dict = [
        {
            "model": conv_model,
            "datasets": (ds_train_conv, ds_test_conv),
            "name": "conv"
        },
        {
            "model": rnn_model,
            "datasets": (ds_train, ds_test),
            "name": "rnn"
        },
        {
            "model": lstm_model,
            "datasets": (ds_train, ds_test),
            "name": "lstm"
        },
        {
            "model": gru_model,
            "datasets": (ds_train, ds_test),
            "name": "gru"
        },
    ]
    return models_dict


def save_json(obj):
    f = open(MODELS_FOLDER+"/history.json","w")
    json.dump(obj, f)
    f.close()

# #show up models
# for md in models:
#     print(md["name"])
#     md["model"].summary()

full_history = {}

for iter in range(REPEAT_COUNT):
    print(f"REPEAT: {str(iter)}")
    full_history[iter] = {}
    models_dict = get_models()

    for md in models_dict:
        id = md["name"]+"_"+str(iter)
        print(f"RUN: {id}")
        model = md["model"]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        train, test = md["datasets"]
        train = train.batch(BACH_SIZE)
        test = test.batch(BACH_SIZE)

        model.fit(
            train,
            validation_data=test,
            epochs=MAX_EPOCHS,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(verbose=1, patience=3),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=MODELS_FOLDER+"/"+id,
                    save_weights_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True)])

        full_history[iter][md["name"]] = copy.deepcopy(model.history.history)
        save_json(full_history)
        
        

