# USAGE 
"""
python mnist_mlp.py \
--input_path "../data/mnist_png" \
--output_path "./" \
--epochs 5 \
--learning_rate 1e-3\
--batch_size 128
"""

# Import the necessary packages
import click
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision import ClassificationInterpretation 
from fastai.vision.transform import get_transforms
from fastai.vision.data import ImageItemList
from fastai.basic_train import Learner
from fastai.metrics import accuracy
from pathlib import Path
from sklearn.metrics import classification_report

## Defining Model
class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 512, bias=True) 
        self.lin2 = nn.Linear(512, 256, bias=True)
        self.lin3 = nn.Linear(256, 10, bias=True)

    def forward(self, xb):
        x = xb.view(-1,784) ## Equivalent to Flatten in Keras 28*28 -> 784
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)

@click.command()
@click.option( 
    '--input_path',
    '-ip',
    default='./',
    required=True,
    help='Path to mnist_png data'
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
    default=5,
    help='Number of epochs'
)

@click.option(
    '--learning_rate',
    '-lr',
    default=1e-3,
    help='Learning Rate'
)

def run_mnist(  input_path, 
                output_path, 
                batch_size,
                epochs,
                learning_rate,
                model=Mnist_NN()):

    path = Path(input_path)

    ## Defining transformation
    ds_tfms = get_transforms(do_flip=False, 
                            flip_vert=False, 
                            max_rotate= 15,
                            max_zoom=1.1, 
                            max_lighting=0.2, 
                            max_warp=0.2)

    ## Creating Databunch
    data = (ImageItemList.from_folder(path, convert_mode='L')
            .split_by_folder(train='training', valid='testing')
            .label_from_folder()
            .transform(tfms=ds_tfms, size=28)
            .databunch(bs=batch_size))



    ## Defining the learner
    mlp_learner = Learner(  data=data,
                            model=model, 
                            loss_func=nn.CrossEntropyLoss(),
                            metrics=accuracy)

    #Training the model
    mlp_learner.fit_one_cycle(epochs,learning_rate)

    val_acc = int(np.round(mlp_learner.recorder.metrics[-1][0].numpy().tolist(),3)*1000)

    ## Saving the model
    mlp_learner.save('mlp_mnist_stg_1_'+str(val_acc))

    ## Evaluation
    print("Evaluating Network..")
    interp = ClassificationInterpretation.from_learner(mlp_learner)
    print(classification_report(interp.y_true,interp.pred_class))

    ## Plotting train and validation loss
    mlp_learner.recorder.plot_losses()
    plt.savefig(output_path+'/loss.png')

    mlp_learner.recorder.plot_metrics()
    plt.savefig(output_path+'/metric.png')

if __name__ == '__main__':
    run_mnist()

