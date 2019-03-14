## Image similarity search using FastAI and Locality Semantic hashing

__High-level approach__ -
1) Transfer learning from a ResNet-34 model(trained on ImageNet) to detect 101 classes in [Caltech-101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) using [FastAI](http://docs.fast.ai/) and [Pytorch](https://pytorch.org/) with 94.7% accuracy.
2) Take the output of second last fully connected layer from trained ResNet 34 model to get embedding for all 9,144 Caltech-101 images.
3) Use Locality Sensitive hashing using [LShash3](https://pypi.org/project/lshash3/) package to create LSH hashing for our image embedding which enables fast approximate nearest neighbor search
4) Then given an image, we can convert it into image embedding using our trained model and then search similar images using Approximate nearest neighbor on Caltech-101 dataset.

__Following is the logical sequence to read the material__ -
1) [Read the Blog Post]()  - [To-Do: Write the blog and update the link.]
2) [Image similarity on Caltech101 using FastAI, Pytorch and Locality Sensitive Hashing.ipynb](https://github.com/aayushmnit/Deep_learning_explorations/blob/master/8_Image_similarity_search/Image%20similarity%20on%20Caltech101%20using%20FastAI%2C%20Pytorch%20and%20Locality%20Sensitive%20Hashing.ipynb) - Contains code to train ResNet-34 model dataset and preprocessing before training. Also contains code to get image embedding and create Locality-sensitive hashing. (Step 1-3)

3) [find_similar_images.py](https://github.com/aayushmnit/Deep_learning_explorations/blob/master/8_Image_similarity_search/find_similar_image.py) - To find similar images from Caltech101 dataset based on any given Image on the internet.

### Output-
#### Example 1 - Pizza
![Similarity Image](https://raw.githubusercontent.com/aayushmnit/Deep_learning_explorations/master/8_Image_similarity_search/output/output.png)

#### Example 2 - Chairs
![Similar images for chairs](https://raw.githubusercontent.com/aayushmnit/Deep_learning_explorations/master/8_Image_similarity_search/output/output1.png)


### Dependencies-
- Python 3.6.7
- numpy 1.15.4
- Pytorch 1.0.0
- fastai 1.0.45
- click 6.7
- opencv 4.0.0
- imutils 0.4.6
- lshash3 0.0.4dev