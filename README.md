# udacity-traffic-sign-classifier
My submission for the traffic sign classifier project.
# **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Documentation/observation_img.png 
[image2]: ./Documentation/observation1.png "Diagram"
[image3]: ./Documentation/normalize.png "Normalize"
[image4]: ./Documentation/translated.png "Translated"
[image5]: ./Documentation/warped.png "Warped"
[image6]: ./Documentation/zoom.png "Zoomed"
[image7]: ./Documentation/augmenting.png "Augmenting"
[image8]: ./Documentation/stacked.png "Stacked Data"
[image9]: ./Documentation/optimum1.png "Accuracy"
[image10]: ./Documentation/my_images.png "My Images"
[image11]: ./Documentation/my_pre.png "Preprocessed"
[image12]: ./Documentation/pred.png "Predictions"
[image13]: ./Documentation/top5.png "Top 5 Predictions"
[image14]: ./Documentation/optimum1.png "Valid Acc"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/BY571/udacity-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set: 

![alt text][image1]

And a bar chart showing how the data is divided in the different classes/labels:

![alt text][image2]

... you can see that distribiution is not optimal. Therefor augmenting the Dataset to at least 1000 images each class (red broken line). So the CNN is has at least 1000 images for each class to train on.

### Pre-processing the Data Set and Augmentation

#### 1. Pre-process
The Preprocessing was done by converting the images to grayscale an normalize them. 

![alt text][image3]


#### 2. Augmenting the Dataset

The augmentation is applied by 3 functions. [Useful cv2 functions](https://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html)

1. random_translate 
![alt text][image4]
2. random_warp
![alt text][image5]
3. random_scale
![alt text][image6]

Some examples of the generated and original images:
![alt text][image7]
After applying the augmentation to the dataset the final distribution is the following:
... as planed >=1000 images each label.
![alt text][image8]

### Design and Test a Model Architecture
#### 1. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 1     	| 1x1 stride, "valid" padding, outputs=28x28x20	|
| RELU					|												|
| Max Pooling	      	| 2x2 stride,"valid" padding,  outputs 14x14x20	|
| Convolution 2 	    | 1x1 stride, "valid" padding, output=10x10x40	|
| RELU          		|           									|
| Max Pooling			| 2x2 stride,"valid" padding, Output = 5x5x40	|
|Flatten				|Input = 5x5x40, Output = 1000					|
|Fully Connected 1		|Input = 1000, Output = 325						|
|RELU           		|			                        			|
|Dropout        		|keep_prob = 0.5                   				|
|Fully Connected 2		|Input = 325, Output = 175						|
|RELU           		|			                        			|
|Dropout        		|keep_prob = 0.5                   				|
|Fully Connected 3		|Input = 175, Output = 43						|


To train the model, I used an batch size of 190 an trained 100 epochs. For the learning rate I chose 0.0009 and I applied dropout with a keep_probability of 0.5. With that, I chose the AdamOptimizier to minimize the loss. Which is used by the softmax_cross_entropy_with_logits function an the reduce_mean function of Tensorflow.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 99.3% 
![alt text][image14]

#### Test set Accuracy:
* test set accuracy of 96.6%


If an iterative approach was chosen:
* First training with RGB images and without augmentation
* Converting to grayscale and normalize the Images
* Applying the Dropout-function and playing around with different keep_prob
* Testing a variety of learning rates / epochs and batch sizes 
* Extending the structure of the CNN and adapting the hyperparameters (learning rate, epochs and batch size)


### Test a Model on New Images


Here are six German traffic signs that I found on the web:

![alt text][image10] 

Preprocessing my test images:

![alt text][image11] 

#### 2. Loading the trained and saved model and testing it with my own images:

Here are the results of the prediction:

![alt text][image12]

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road      		| Bumpy road                 (22)				| 
| No passing   			| No passing                 (9)				|
| Yield					| Yield	                     (13)    			|
| Children crossing		| Children crossing          (28)				|
| Speed limit (30km/h)	| Speed limit (30km/h)       (1)  				|
| Slippery road     	| Slippery road              (23)  				|

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 

Subsequent you can see the probability upon what my model predicted the correct traffic sign.
You can see that for 5 of the six signs my model has a 100% sureness of predicting the right traffic sign. Only for the Speed limit (30km/h) it has a sureness of ~ 70%
![alt text][image13]

