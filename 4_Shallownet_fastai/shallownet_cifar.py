# USAGE
"""
python shallownet_cifar.py \
--input_path "../data/cifar10" \
--output_path "./" \
--epochs 5 \
--learning_rate 1e-2\
--batch_size 64
"""

# Import the necessary packages
import click
import numpy as np
import matplotlib.pyplot as plt
from shallownet import *
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
def run_shallownet(input_path,
              output_path,
              batch_size,
              epochs,
              learning_rate):

    path = Path(input_path)

    # Creating Databunch
    data = (ImageItemList.from_folder(path)
            .split_by_folder(train='train', valid='test')
            .label_from_folder()
            .transform(tfms=None, size=32)
            .databunch(bs=batch_size))

    # Defining the learner
    sn_learner = Learner(data=data, 
                      model=ShallowNet(n_class=data.c, 
                                       size=32, 
                                       in_channels=3), 
                      loss_func=nn.CrossEntropyLoss(),
                      metrics=accuracy)

    # Training the model
    sn_learner.fit_one_cycle(epochs, learning_rate)

    val_acc = int(
        np.round(sn_learner.recorder.metrics[-1][0].numpy().tolist(), 3)*1000)

    # Saving the model
    sn_learner.save('sn_cifar10_stg_1_'+str(val_acc))

    # Evaluation
    print("Evaluating Network..")
    interp = ClassificationInterpretation.from_learner(sn_learner)
    print(classification_report(interp.y_true, interp.pred_class))

    # Plotting train and validation loss
    sn_learner.recorder.plot_losses()
    plt.savefig(output_path+'/loss.png')

    sn_learner.recorder.plot_metrics()
    plt.savefig(output_path+'/metric.png')


if __name__ == '__main__':
    run_shallownet()
