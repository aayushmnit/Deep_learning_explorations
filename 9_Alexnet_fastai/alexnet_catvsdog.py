# USAGE
"""
python alexnet_catsvsdog.py \
--input_path "../data/cats_dog" \
--output_path "./" \
--epochs 10 \
--learning_rate 1e-3\
--batch_size 128 \
"""

# Import the necessary packages
import click
import numpy as np
import matplotlib.pyplot as plt
from alexnet import *
from fastai.vision import ClassificationInterpretation
from fastai.vision.data import ImageDataBunch
from fastai.vision.data import get_image_files
from fastai.basic_train import Learner
from fastai.metrics import accuracy
from pathlib import Path
from sklearn.metrics import classification_report


@click.command()
@click.option(
    "--input_path", "-ip", default="./", required=True, help="Path to CIFAR10 data"
)
@click.option(
    "--output_path",
    "-op",
    default="./",
    required=True,
    help="path to the output loss/accuracy plot",
)
@click.option("--batch_size", "-bs", default=64, help="Batch size")
@click.option("--epochs", "-e", default=10, help="Number of epochs")
@click.option("--learning_rate", "-lr", default=1e-2, help="Learning Rate")
def run_alexnet(input_path, output_path, batch_size, epochs, learning_rate, batch_norm):
    # Load image databunch
    print("[INFO] Loading Data")
    data = load_catsvsdog(input_path)

    # Defining the learner
    alexnet_learner = Learner(
        data=data,
        model=ALEXNet(n_class=data.c),
        loss_func=nn.CrossEntropyLoss(),
        metrics=accuracy,
    )

    # Training the model
    print("[INFO] Training started.")
    alexnet_learner.fit_one_cycle(epochs, learning_rate)

    # Validation accuracy
    val_acc = int(
        np.round(alexnet_learner.recorder.metrics[-1][0].numpy().tolist(), 3) * 1000
    )

    # Saving the model
    print("[INFO] Saving model weights.")
    alexnet_learner.save("alexnet_catsvsdog_stg_1_" + str(val_acc))

    # Evaluation
    print("[INFO] Evaluating Network.")
    evaluate_model(alexnet_learner, output_path, plot=True)


def load_catsvsdog(input_path):
    """
    Function to load data from cats vs dog Kaggle competition
    """
    path = Path(input_path)
    fnames = get_image_files(path)

    # Creating Databunch
    data = ImageDataBunch.from_name_re(
        path,
        fnames,
        pat=r"([^/]+)\.\d+.jpg$",
        ds_tfms=get_transforms(),
        valid_pct=0.2,
        size=227,
        bs=bs,
    ).normalize()

    return data


def evaluate_model(model, output_path, plot=True):
    """
    Function to evaluate model performance.
    Generates Classification report, loss and metric plot.
    """
    interp = ClassificationInterpretation.from_learner(model)
    print(classification_report(interp.y_true, interp.pred_class))

    if plot:
        # Plotting loss progression with each epoch
        model.recorder.plot_losses()
        plt.savefig(output_path + "/loss.png")

        # Plotting metric progression with each epoch
        model.recorder.plot_metrics()
        plt.savefig(output_path + "/metric.png")


if __name__ == "__main__":
    run_alexnet()
