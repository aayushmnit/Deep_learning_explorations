# USAGE
"""
python lenet_mnist.py \
--input_path "../data/mnist_png" \
--output_path "./" \
--epochs 5 \
--learning_rate 1e-2\
--batch_size 64
"""

# Import the necessary packages
import click
import numpy as np
import matplotlib.pyplot as plt
from lenet import *
from fastai.vision.transform import get_transforms
from fastai.vision.data import ImageItemList
from fastai.basic_train import Learner
from fastai.metrics import accuracy
from fastai.vision import ClassificationInterpretation
from pathlib import Path
from sklearn.metrics import classification_report


@click.command()
@click.option(
    "--input_path", "-ip", default="./", required=True, help="Path to MNIST data"
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
def run_lenet(input_path, output_path, batch_size, epochs, learning_rate):

    path = Path(input_path)

    # Creating Databunch
    ds_tfms = get_transforms(
        do_flip=False,
        flip_vert=False,
        max_rotate=15,
        max_zoom=1.1,
        max_lighting=0.2,
        max_warp=0.2,
    )

    data = (
        ImageItemList.from_folder(path, convert_mode="L")
        .split_by_folder(train="training", valid="testing")
        .label_from_folder()
        .transform(tfms=ds_tfms, size=28)
        .databunch(bs=batch_size)
    )

    # Defining the learner
    lenet_learner = Learner(
        data=data,
        model=LeNet(n_class=data.c, size=28, in_channels=1),
        loss_func=nn.CrossEntropyLoss(),
        metrics=accuracy,
    )

    # Training the model
    lenet_learner.fit_one_cycle(epochs, learning_rate)

    val_acc = int(
        np.round(lenet_learner.recorder.metrics[-1][0].numpy().tolist(), 3) * 1000
    )

    # Saving the model
    lenet_learner.save("lenet_mnist_stg_1_" + str(val_acc))

    # Evaluation
    print("Evaluating Network..")
    interp = ClassificationInterpretation.from_learner(lenet_learner)
    print(classification_report(interp.y_true, interp.pred_class))

    # Plotting train and validation loss
    lenet_learner.recorder.plot_losses()
    plt.savefig(output_path + "/loss.png")

    lenet_learner.recorder.plot_metrics()
    plt.savefig(output_path + "/metric.png")


if __name__ == "__main__":
    run_lenet()
