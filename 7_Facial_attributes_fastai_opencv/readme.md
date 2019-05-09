## Facial attributes detection using FastAI and Pytorch

In this folder, I have trained a model to detect 40 facial attributes using [CelebA dataset](https://www.kaggle.com/jessicali9530/celeba-dataset) from kaggle using FastAi and Pytorch.

Following is the logical sequence to follow the material -
1) [Read the Blog Post](https://towardsdatascience.com/real-time-multi-facial-attribute-detection-using-transfer-learning-and-haar-cascades-with-fastai-47ff59e36df0) 
2) [Data_prepration.ipnb](https://nbviewer.jupyter.org/github/aayushmnit/Deep_learning_explorations/blob/master/7_Facial_attributes_fastai_opencv/Data_prepration.ipynb) - Contains code to download dataset and preprocessing before training
3) [MultiClass classification on CelebA dataset using FastAI.ipynb](https://nbviewer.jupyter.org/github/aayushmnit/Deep_learning_explorations/blob/master/7_Facial_attributes_fastai_opencv/MultiClass%20classification%20on%20CelebA%20dataset%20using%20FastAI.ipynb) - For training the multi-class facial attribute model
4) [detect_features.py](https://github.com/aayushmnit/Deep_learning_explorations/blob/master/7_Facial_attributes_fastai_opencv/detect_features.py) - To run model using web cam.

## Output
![Output GIF](https://github.com/aayushmnit/Deep_learning_explorations/blob/master/7_Facial_attributes_fastai_opencv/output.gif?raw=true)

## Dependencies
- numpy
- Pytorch
- fastai v1.0.45
- click
- opencv2