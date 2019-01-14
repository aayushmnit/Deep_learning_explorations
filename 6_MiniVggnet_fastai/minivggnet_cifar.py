# USAGE
"""
python minivggnet_cifar.py \
--input_path "../data/cifar10" \
--output_path "./" \
--epochs 10 \
--learning_rate 1e-3\
--batch_size 64 \
--batch_norm True
"""

# Import the necessary packages
import click
import numpy as np
import matplotlib.pyplot as plt
from minivggnet import *
from fastai.vision import ClassificationInterpretation
from fastai.vision.data import ImageItemList
from fastai.basic_train import Learner
from fastai.metrics import accuracy
from pathlib import Path
from sklearn.metrics import classification_report


@click.command()
@click.option(
    '--input_path',
    '-ip',
    default='./',
    required=True,
    help='Path to CIFAR10 data'
)
@click.option(
    '--output_path',
    '-op',
    default='./',
    required=True,
    help='path to the output loss/accuracy plot'
)
@click.option(
    '--batch_size',
    '-bs',
    default=64,
    help='Batch size'
)
@click.option(
    '--epochs',
    '-e',
    default=10,
    help='Number of epochs'
)
@click.option(
    '--learning_rate',
    '-lr',
    default=1e-2,
    help='Learning Rate'
)
@click.option(
    '--batch_norm',
    '-bn',
    default=True,
    type=bool,
    help='Batch normalization'
)
def run_minivggnet(input_path,
                   output_path,
                   batch_size,
                   epochs,
                   learning_rate,
                   batch_norm):

    path = Path(input_path)

    # Creating Databunch
    data = (ImageItemList.from_folder(path)
            .split_by_folder(train='train', valid='test')
            .label_from_folder()
            .transform(tfms=None, size=32)
            .databunch(bs=batch_size))

    # Defining the learner
    minivgg_learner = Learner(data=data,
                              model=MiniVGGNet(
                                  batch_norm=batch_norm,
                                  size=32,
                                  n_class=data.c),
                              loss_func=nn.CrossEntropyLoss(),
                              metrics=accuracy)

    # Training the model
    minivgg_learner.fit_one_cycle(epochs, learning_rate)

    val_acc = int(
        np.round(minivgg_learner.recorder.metrics[-1][0].numpy().tolist(), 3)*1000)

    # Saving the model
    minivgg_learner.save('minivgg_cifar10_stg_1_'+str(val_acc))

    # Evaluation
    print("Evaluating Network..")
    interp = ClassificationInterpretation.from_learner(minivgg_learner)
    print(classification_report(interp.y_true, interp.pred_class))

    # Plotting train and validation loss
    minivgg_learner.recorder.plot_losses()
    plt.savefig(output_path+'/loss.png')

    minivgg_learner.recorder.plot_metrics()
    plt.savefig(output_path+'/metric.png')


if __name__ == '__main__':
    run_minivggnet()
