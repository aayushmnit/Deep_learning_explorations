## Multi-Layer Perceptron using FastAI and Pytorch

In this folder, I have trained Multi-Layer Perceptron using FastAI and Pytorch. To go through the coding process, you can refer to the [Jupyter notebook](https://nbviewer.jupyter.org/github/aayushmnit/Deep_learning_explorations/blob/master/3_MLP_FastAI/MLP%20using%20Fast%20%20%26%20Pytorch.ipynb). 

Furthermore, I have used [Click](https://click.palletsprojects.com/en/7.x/) python package to create a command line executable script. Click makes it easy to create a CLI API by simplifying code of the implementation of error handling and also helps in creating an automatic help page.

You can use the following example command with different parameters to experiment with this script-
```bash
python mnist_mlp.py --input_path "../data/mnist_png" --output_path "./" --epochs 5 --learning_rate 1e-3 --batch_size 128
```

Here's how it looks when it run in command interface - 
<img scr="https://raw.githubusercontent.com/aayushmnit/Deep_learning_explorations/master/3_MLP_FastAI/run_snapshot.PNG">


Outputs are loss and metric plots during training -
<img src="https://raw.githubusercontent.com/aayushmnit/Deep_learning_explorations/master/3_MLP_FastAI/loss.png">

<img src="https://raw.githubusercontent.com/aayushmnit/Deep_learning_explorations/master/3_MLP_FastAI/metric.png">

#### Help interface of Click - 
<img src="https://raw.githubusercontent.com/aayushmnit/Deep_learning_explorations/master/3_MLP_FastAI/help_snapshot.PNG">
