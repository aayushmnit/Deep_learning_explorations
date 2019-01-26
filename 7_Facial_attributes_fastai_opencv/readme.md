## Facial attributes detection using FastAI and Pytorch

In this folder, I have trained a model to detect 40 facial attributes using [CelebA dataset](https://www.kaggle.com/jessicali9530/celeba-dataset) from kaggle using FastAi and Pytorch.

Following is the logical sequence to follow the code -
1) [Data_prepration.ipnb](https://nbviewer.jupyter.org/github/aayushmnit/Deep_learning_explorations/blob/master/7_Facial_attributes_fastai_opencv/Data_prepration.ipynb) - Contains code to download dataset and preprocessing before training
2) [MultiClass classification on CelebA dataset using FastAI.ipynb](https://nbviewer.jupyter.org/github/aayushmnit/Deep_learning_explorations/blob/master/7_Facial_attributes_fastai_opencv/MultiClass%20classification%20on%20CelebA%20dataset%20using%20FastAI.ipynb) - For training the multi-class facial attribute model
3) [detect_features.py](https://github.com/aayushmnit/Deep_learning_explorations/blob/master/7_Facial_attributes_fastai_opencv/detect_features.py) - To run model using web cam.

## Output
![Output GIF](https://github.com/aayushmnit/Deep_learning_explorations/blob/master/7_Facial_attributes_fastai_opencv/output.gif?raw=true)

## Dependencies
- numpy
- Pytorch
- fastai v1
- click
- opencv2