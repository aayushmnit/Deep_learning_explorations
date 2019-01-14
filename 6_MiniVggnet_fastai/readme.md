## Shallownet on CIFAR10 using FastAI and Pytorch

In this folder, I have trained a miniature version of VGG architecture using FastAI and Pytorch. To go through the coding process, you can refer to the [Jupyter notebook](https://nbviewer.jupyter.org/github/aayushmnit/Deep_learning_explorations/blob/master/6_MiniVggnet_fastai/MiniVggNet%20using%20FastAI.ipynb). Before running the code make sure you download and unzip [CIFAR10 dataset](https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz) from [fastai dataset](https://course.fast.ai/datasets) pages

You can use the following example command with different parameters to experiment with this script-
```bash
python minivggnet_cifar.py --input_path "../data/cifar10" --output_path "./" --epochs 10 --learning_rate 1e-3 --batch_size 64 --batch_norm True
```

Here's how it looks when it run in command interface - 
```
epoch     train_loss  valid_loss  accuracy
1         1.364085    1.214785    0.560100
2         1.041337    0.888044    0.689800
3         0.899581    0.863549    0.698700
4         0.777371    0.691828    0.759100
5         0.649673    0.639749    0.781300
6         0.523551    0.633896    0.793100
7         0.429066    0.546421    0.818200
8         0.330242    0.532833    0.828700
9         0.295064    0.520767    0.832400
10        0.264399    0.525779    0.831600
Evaluating Network..
              precision    recall  f1-score   support

           0       0.85      0.85      0.85      1000
           1       0.93      0.91      0.92      1000
           2       0.78      0.70      0.74      1000
           3       0.70      0.67      0.68      1000
           4       0.78      0.83      0.81      1000
           5       0.76      0.78      0.77      1000
           6       0.84      0.90      0.87      1000
           7       0.88      0.87      0.87      1000
           8       0.91      0.91      0.91      1000
           9       0.88      0.91      0.89      1000

   micro avg       0.83      0.83      0.83     10000
   macro avg       0.83      0.83      0.83     10000
weighted avg       0.83      0.83      0.83     10000
```

## Dependencies
- numpy
- scikit-learn
- Pytorch
- fastai v1
- click
