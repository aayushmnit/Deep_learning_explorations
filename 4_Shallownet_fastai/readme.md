## Shallownet on CIFAR10 using FastAI and Pytorch

In this folder, I have trained ShallowNet using FastAI and Pytorch. To go through the coding process, you can refer to the [Jupyter notebook](https://nbviewer.jupyter.org/github/aayushmnit/Deep_learning_explorations/blob/master/4_Shallownet_fastai/Shallownet%20using%20FastAI.ipynb). Before running the code make sure you download and unzip [CIFAR10 dataset](https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz) from [fastai dataset](https://course.fast.ai/datasets) pages

You can use the following example command with different parameters to experiment with this script-
```bash
>python shallownet_cifar.py --input_path "../data/cifar10" --output_path "./" --epochs 5 --learning_rate 1e-2 --batch_size 64
```

Here's how it looks when it run in command interface - 
```
epoch     train_loss  valid_loss  accuracy
1         1.587146    1.556950    0.458700
2         1.551461    1.611428    0.417600
3         1.351812    1.463269    0.484700
4         1.052478    1.303436    0.565400
5         0.853768    1.308465    0.570900

Evaluating Network..
              precision    recall  f1-score   support

           0       0.61      0.60      0.61      1000
           1       0.72      0.69      0.71      1000
           2       0.41      0.41      0.41      1000
           3       0.40      0.36      0.38      1000
           4       0.48      0.53      0.50      1000
           5       0.45      0.45      0.45      1000
           6       0.67      0.68      0.68      1000
           7       0.62      0.61      0.62      1000
           8       0.67      0.71      0.69      1000
           9       0.65      0.67      0.66      1000

   micro avg       0.57      0.57      0.57     10000
   macro avg       0.57      0.57      0.57     10000
weighted avg       0.57      0.57      0.57     10000
```

## Dependencies
- numpy
- scikit-learn
- Pytorch
- fastai v1
- click
