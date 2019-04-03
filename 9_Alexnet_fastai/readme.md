## AlexNet on Cats vs Dog using FastAI and Pytorch

In this folder, I have trained AlexNet model from scratch using Pytorch and FastAI. To go through the coding process, you can refer to the [Jupyter notebook](https://nbviewer.jupyter.org/github/aayushmnit/Deep_learning_explorations/blob/master/9_Alexnet_fastai/AlexNet%20using%20FastAI.ipynb). Before running the code make sure you download and unzip train folder of [Dogs vs. Cats challenge](https://www.kaggle.com/c/dogs-vs-cats/data) from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data) and rename the folder as cats_dog.

You can use the following example command with different parameters to experiment with this script-
```bash
python alexnet_catvsdog.py --input_path "../data/cats_dog" --output_path "./" --epochs 10 --learning_rate 1e-3 --batch_size 128
```

Here's how it looks when it run in command interface - 
```
epoch	   train_loss	   valid_loss	   accuracy
1	   0.638005	   0.766830	   0.655800
2	   0.534617	   0.494307	   0.768000
3	   0.477765	   0.535594	   0.730800
4	   0.398305	   0.414883	   0.796400
5	   0.344727	   0.398703	   0.838800
6	   0.293345	   0.304188	   0.878400
7	   0.242300	   0.204189	   0.916600
8	   0.212505	   0.194056	   0.923600
9	   0.173101	   0.159934	   0.937600
10	   0.160533	   0.157122	   0.937600
Evaluating Network..
            precision    recall  f1-score   support

           0       0.95      0.93      0.94      2503
           1       0.94      0.95      0.94      2497

   micro avg       0.94      0.94      0.94      5000
   macro avg       0.94      0.94      0.94      5000
weighted avg       0.94      0.94      0.94      5000
```

Loss plot(Cross-entropy)-
![Loss Plot](https://raw.githubusercontent.com/aayushmnit/Deep_learning_explorations/master/9_Alexnet_fastai/loss.png)

Metric plot(Accuracy) -
![Metric Plot](https://raw.githubusercontent.com/aayushmnit/Deep_learning_explorations/master/9_Alexnet_fastai/metric.png)

## Dependencies
- numpy v1.15.4
- scikit-learn v0.20.1
- pytorch v1.0
- fastai v1.0.45
- click v6.7
