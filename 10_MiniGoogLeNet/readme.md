## MiniGoogleNet on CIFAR10 using FastAI and Pytorch

In this folder, I have trained a miniature version of Inception architecture using FastAI and Pytorch. To go through the coding process, you can refer to the [Jupyter notebook](https://nbviewer.jupyter.org/github/aayushmnit/Deep_learning_explorations/blob/master/10_MiniGoogLeNet/MiniGoogLeNet%20using%20FastAI.ipynb). Before running the code make sure you download and unzip [CIFAR10 dataset](https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz) from [fastai dataset](https://course.fast.ai/datasets) pages

You can use the following example command with different parameters to experiment with this script-
```bash
python minigooglenet_cifar.py --input_path "../data/cifar10" --output_path "./" --epochs 10 --learning_rate 5e-3 --batch_size 128 
```

Here's how it looks when it run in command interface - 
```
epoch     train_loss  valid_loss  accuracy  time
0         1.097950    1.561092    0.481700  04:21
1         0.867719    1.234036    0.615100  03:49
2         0.694231    0.806185    0.727200  03:49
3         0.551282    0.758939    0.745300  03:43
4         0.456713    0.584816    0.804400  03:45
5         0.351192    0.466134    0.847500  03:48
6         0.238730    0.409035    0.865700  03:43
7         0.144988    0.391103    0.881400  03:40
8         0.070785    0.394344    0.887300  03:48
9         0.044012    0.393803    0.888900  03:42
Evaluating Network..
              precision    recall  f1-score   support

           0       0.88      0.90      0.89      1000
           1       0.96      0.94      0.95      1000
           2       0.87      0.82      0.85      1000
           3       0.78      0.78      0.78      1000
           4       0.86      0.89      0.87      1000
           5       0.84      0.83      0.84      1000
           6       0.90      0.93      0.92      1000
           7       0.92      0.92      0.92      1000
           8       0.94      0.94      0.94      1000
           9       0.92      0.93      0.92      1000

   micro avg       0.89      0.89      0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
```

## Dependencies
- numpy
- scikit-learn
- Pytorch 1.0.0
- fastai v1.0.55
- click
