import torch
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.contrib.handlers.tensorboard_logger import *

from AUC import AUC


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"        # bez tego wysadza kernel kiedy rysuje obrazek
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # potrzebne dla deterministycznego działania


def get_loaders_single_model(batch_size):
    train_data = datasets.CIFAR10("./cifar10", download=True, transform=transforms.ToTensor(), train=True)
    test_data = datasets.CIFAR10("./cifar10", download=True, transform=transforms.ToTensor(), train=False)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader


def get_loaders_committee(batch_size, committee_size):
    train_data = datasets.CIFAR10("./cifar10", download=True, transform=transforms.ToTensor(), train=True)
    test_data = datasets.CIFAR10("./cifar10", download=True, transform=transforms.ToTensor(), train=False)

    train_loaders = [DataLoader(train_data, batch_size=batch_size, sampler=RandomSampler(train_data, True, len(train_data))) for _ in range(committee_size)]
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loaders, test_loader

def augmentation(image):
    transform_type = int(torch.randint(1, 6, (1, 1)))
    if transform_type == 1:
        angle = int(torch.randint(1, 180, (1, 1)))
        image = TF.rotate(image, angle)
    elif transform_type == 2:
        image = TF.hflip(image)
    elif transform_type == 3:
        image = TF.vflip(image)
    elif transform_type == 4:
        image = TF.gaussian_blur(image, (3, 3), (1.5, 1.5))
    elif transform_type == 5:
        image = TF.equalize(image)
    return TF.to_tensor(image)

def get_loaders_augmented(batch_size):
    train_data = datasets.CIFAR10("./cifar10", download=True, transform=transforms.ToTensor(), train=True)
    test_data = datasets.CIFAR10("./cifar10", download=True, transform=transforms.ToTensor(), train=False)

    train_data_augmented = datasets.CIFAR10("./cifar10", download=True, transform=augmentation, train=True)
    train_data_concat = torch.utils.data.ConcatDataset([train_data,train_data_augmented])

    train_loader = DataLoader(train_data_concat, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader


classes = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device used is {device}")


class Committee(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        predictions = torch.stack([model.forward(x).argmax(1) for model in self.models])
        predictions = torch.stack([predictions[:, i].to("cpu").bincount(minlength=10).to(device) for i in range(x.shape[0])]).float()
        return predictions


def plot_confusion_matrix(cm, draw=True, save=False, savefile=""):
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    if draw:
        plt.show()
    if save and savefile and savefile.strip():
        plt.savefig(f"plots/{savefile}.png")

def softmax_output_transform(output):
    y_pred, y = output
    y_pred = torch.softmax(y_pred, 1)
    return y_pred, y

def to_cpy_output_transform(output):
    y_pred, y = output
    return y_pred.to("cpu"), y.to("cpu")

def create_tensorboard_logger(savefile, run_id, trainer, train_evaluator, val_evaluator, model):
    tb_logger = TensorboardLogger(log_dir=f"tb-logger/{run_id}/{savefile}")

    layout = {
        "multiline": {
            "loss": ["Multiline", ["training/loss", "validation/loss"]],
            "accuracy": ["Multiline", ["training/accuracy", "validation/accuracy"]],
            "auc": ["Multiline", ["training/auc", "validation/auc"]],
        },
    }
    tb_logger.add_custom_scalars(layout)

    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=["accuracy", "loss", "auc"],
            global_step_transform=global_step_from_engine(trainer),
        )

    # Attach the logger to the trainer to log model's weights norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=WeightsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsHistHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=GradsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=GradsHistHandler(model)
    )
        
    return tb_logger

def run_model(model, train_loader, test_loader, device=device, draw=True, save=False, savefile="", run_id=""):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)

    val_metrics = {
        "accuracy": Accuracy(device=device),
        "loss": Loss(loss_fn, device=device),
        "auc": AUC(softmax_output_transform, device=device),
        "confusion_matrix": ConfusionMatrix(10, output_transform=to_cpy_output_transform)
    }

    train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy'] * 100:.2f}%, Avg loss: {metrics['loss']:.2f}, AUC: {metrics['auc']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(test_loader)
        metrics = val_evaluator.state.metrics
        # print(metrics["confusion_matrix"])
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy'] * 100:.2f}%, Avg loss: {metrics['loss']:.2f}, AUC: {metrics['auc']:.2f}")

    def score_function(engine):
        metrics = engine.state.metrics
        return metrics["accuracy"]
    
    val_evaluator.add_event_handler(Events.COMPLETED, EarlyStopping(3, score_function, trainer))

    with create_tensorboard_logger(savefile, run_id, trainer, train_evaluator, val_evaluator, model) as tb_logger:
        trainer.run(train_loader, max_epochs=100)

    plot_confusion_matrix(val_evaluator.state.metrics["confusion_matrix"], draw, save, savefile)

def run_models(models, train_loaders, test_loader, device=device, draw=True, save=False, savefile="", run_id=""):
    assert len(models) == len(train_loaders), "Number of models and number of train_loaders should be equal"

    for i, model in enumerate(models):
        print(f"Training model #{i + 1}...")

        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = create_supervised_trainer(model, optimizer, loss_fn, device)

        val_metrics = {
            "accuracy": Accuracy(device=device),
            "loss": Loss(loss_fn, device=device),
            "auc": AUC(softmax_output_transform, device=device)
        }

        train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
        val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            train_evaluator.run(train_loaders[i])
            metrics = train_evaluator.state.metrics
            print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy'] * 100:.2f}%, Avg loss: {metrics['loss']:.2f}, AUC: {metrics['auc']:.2f}")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            val_evaluator.run(test_loader)
            metrics = val_evaluator.state.metrics
            print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy'] * 100:.2f}%, Avg loss: {metrics['loss']:.2f}, AUC: {metrics['auc']:.2f}")

        def score_function(engine):
            metrics = engine.state.metrics
            return metrics["accuracy"]


        val_evaluator.add_event_handler(Events.COMPLETED, EarlyStopping(3, score_function, trainer))
        with create_tensorboard_logger(savefile+"/"+str(i), run_id, trainer, train_evaluator, val_evaluator, model) as tb_logger:
            trainer.run(train_loaders[i], max_epochs=100)


    committee = Committee(models)
    
    loss_fn = nn.CrossEntropyLoss()
    val_metrics = {
        "accuracy": Accuracy(device=device),
        "loss": Loss(loss_fn, device=device),
        "auc": AUC(softmax_output_transform, device=device),
        "confusion_matrix": ConfusionMatrix(10, device=device, output_transform=to_cpy_output_transform)
    }
    evaluator = create_supervised_evaluator(committee, metrics=val_metrics, device=device)

    @evaluator.on(Events.COMPLETED)
    def log_committee_results(evaluator):
        metrics = evaluator.state.metrics
        print(f"Validation Results - Committee Avg accuracy: {metrics['accuracy'] * 100:.2f}%, Avg loss: {metrics['loss']:.2f}, AUC: {metrics['auc']:.2f}")

    evaluator.run(test_loader)
    plot_confusion_matrix(evaluator.state.metrics["confusion_matrix"], draw, save, savefile)
