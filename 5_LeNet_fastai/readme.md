## Shallownet on CIFAR10 using FastAI and Pytorch

In this folder, I have trained ShallowNet using FastAI and Pytorch. To go through the coding process, you can refer to the [Jupyter notebook](). Before running the code make sure you download and unzip [MNIST dataset](https://s3.amazonaws.com/fast-ai-imageclas/mnist_png.tgz) from [fastai dataset](https://course.fast.ai/datasets) pages

You can use the following example command with different parameters to experiment with this script-
```bash
python lenet_mnist.py --input_path "../data/mnist_png" --output_path "./" --epochs 5 --learning_rate 1e-2 --batch_size 64
```

Here's how it looks when it run in command interface - 
```
epoch     train_loss  valid_loss  accuracy
1         0.165560    0.068876    0.977300
2         0.119575    0.072090    0.979400
3         0.081046    0.045853    0.986900
4         0.047947    0.025536    0.992100
5         0.045332    0.019428    0.994200
Evaluating Network..
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       980
           1       0.99      1.00      1.00      1135
           2       0.99      0.99      0.99      1032
           3       0.99      1.00      0.99      1010
           4       0.99      1.00      1.00       982
           5       1.00      0.99      0.99       892
           6       0.99      0.99      0.99       958
           7       0.99      0.99      0.99      1028
           8       0.99      1.00      0.99       974
           9       1.00      0.99      0.99      1009

   micro avg       0.99      0.99      0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
```

## Dependencies
- numpy
- scikit-learn
- Pytorch
- fastai v1
- click
