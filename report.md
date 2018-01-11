# **Traffic Sign Recognition** 


## Overview

### Problem description

In this project we built and trained a fully supervised deep convolutional neural network
that learns the classifier for RGB 32x32 pixel images 
 from German traffic signs dataset. 


### Goals


The accuracy evaluated on validation set was used to measure the model performance.
As suggested in the project guideline, we aimed to get higher than 93% validation accuracy. 


---


## Dataset Summary & Exploration


The original dataset was from [German Traffic Sign Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
We began with the pickled [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
 prepared by Udacity.


###  Dataset summary statistics

Numpy library was used to calculate summary statistics of the traffic
signs data set. The dataset contains three mutually exclusive subsets: train, valid, and test.
The train set includes a total number of 34,799 examples; the valid set contains
4,410 examples; and lastly, the test set contains 12,630 exampels. Each example image
had three color channels and was standardized to 32 * 32 pixels. Thus 
each image data had the shape of (32, 32, 3). 
 
The distribution of image intensities seemed very similar across three subsets -
minimum value 0 and maximum value 255. Also, the mean values and 
standard deviations across the subsets were very close. 

All examples had the corresponding class labels (No missing labels). 
The class labels were 
[numerically encoded ranging from 0 to 42](https://drive.google.com/file/d/1pwdU3dqx9rqrhlSxf2RBAjqzZncyE6tJ/view?usp=sharing). All three 
 subsets had a total of 43 unique class labels. 

|Dataset | Num. of examples| Num. of classes| Image shape| Min | Max | Mean | StdDev|
|--------|-------------------| ------------------------|------------|-----|-----|------|-------|
| Train  | 34,799            | 43                      |32 x 32 x 3 |  0  | 255 | 82.6775 | 67.8509|
| Valid  | 4, 410            | 43                      |32 x 32 x 3 | 0   | 255 | 83.5564 | 69.8877|
| Test   | 12,630            | 43                      |32 x 32 x 3 |   0 | 255 | 82.1484 | 68.7441|




###  Exploratory visualization

The image below shows sample images from the train dataset.
The mean intensity and contrast(difference between brightest pixel and 
darkest pixel) vary across the samples.

{{< figure src="/img/traffic_signs/traffic_sign_grid_small.png" 
    title="Sample images from train dataset" >}}

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/traffic_sign_grid_small.png)


There was imbalance among the classes; 
the label data across the classes is not evenly distributed. 
For the training dataset, a total of 15 classes have less than 1% 
of the total examples, while seven labels are more frequent 
than 5%. Meanwhile, 
 approximately the distributions look quite similar to each other,
suggesting that possibly all three data subsets 
 come from the same data generating 
 distribution.

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/label_dist_small.png)


## Preprocessing the image data

We applied four preprocessing techniques for the image data: 
converting to grayscale, 
global contrast normalization, MinMax normalization, and dataset augmentation.

In order for the training not to be affected by the order of its labels, 
we shuffled each dataset before normalization.  

#### 1. Converting to grayscale
In [Sermanet and Lecun(2010)](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), it was reported that grayscale image data 
performed better than the color images. In order to check if the same is 
applied to our dataset, we also prepared gray image datasets. 

#### 2. Global contrast normalization (GCN)

As we observed in the data exploration step, images had varying 
level of contrast. In order to safely remove these variations, the
global contrast normalization was applied to each image. 



#### 3. MinMax normalization

Numerical optimization on inputs with varying ranges can result in failure.
Thus the intensity in each image data was standardized so that 
te value all lie in the range \[0, 1]. 


Below shows the example that 
preprocessing helped improve image quality significantly. 
We MinMax normalized two version of training datasets, grayscaled + GNC and 
RGB + GNC, and proceeded to augmentation for both.  

{{< figure src="/img/traffic_signs/preprocessing.png" 
    title="Sample image after each preprocessing step" >}}

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/label_dist_small.png)


#### 4. Data augmentation
##### 1) Random image transformation
 

Since the convolutional neural network is not invariant to
some image operations such as rotation and scaling,
the task of object detection can greatly benefit from data augmentation. 
Artificially boosting the size of the training set can prevent 
the model from overfitting, effectively working as regualarization.  

After training the baseline network architecture (LeNet5), 
we found that the model underfits the data. 
We increased the model capacity by adding additional 
convolutional layers, 
increase the kernel size, and adjusting other hyperparameters,
which resulted in turn overfitting. 
To curb the overfitting, we boosted the example size
by double, then quadruple until the overfitting stops. 
 
 
We randomly applied simple image operations randomly on both gray and RGB images 
in the training set, including rotation (\[-15, +15] degress), 
shifting (up to 3 pixels vertically and horizontally), 
zooming(sclale down up to 90% or blow up up to 110%), and 
shearing(up to 7 pixels). 

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/image_transform_operation.png)

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/augmented_examples_2.png)

  

##### 2) Consideration for class imbalance

In the data exploration step, the class imbalance was observed in all three 
data sets. If we augment the dataset preserving the relative ratio of labels, 
then the class imbalance is amplified. In this case, the algorithm may
 simply classify everything as belonging to the majority class. 
 In order to prepare more balanced dataset, we designed the 
 number of augmentation for each image to be dependent on not only the 
*scale factor* but also  *weights_on_scarce*. The weights_on_scarce
parameter decides how much we boost the image in case it belongs to the 
 minority class. 
 
 
 ```python
n_transform = np.floor((1 - weights_on_scarce * train_freq_normalized) * s)
```

Below shows how the class imbalance was improved when we set
the scale_factor = 4 and the weight_on_scarce = 0.75.
The max frequency / min frequency was decreased from 11.16 to 4.82.


![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/class_imbalance_reduced_2.png)


After the preprocessing steps mentioned above applied, 
two augmented training sets were prepared.

| Dataset | Num. of examples | Num. of classes       | Image shape | Min | Max | Mean  | StdDev |
|---------|--------------------|-------------------------|------------ |-----|-----|------ |--------|
| Train_augmented (gray)   | 99,176         |              43         |32 x 32 x 3  |  0  | 1   | 0.3411| 0.2960 | 
| Train_augmented (rgb)   | 99,176            |              43         |32 x 32 x 3  |  0  | 1   | 0.3476| 0.2885 | 



---

## Training details - Model Architecture for classification with CNN

#### 1. Training details

The crossentropy loss function was used for multiclass classification. 
Adam (*Adaptive moment estimation*) optimizer ($\beta_1 = 0.9, \beta_2 = 0.999$) 
with minibatch size 128 was used to
update the parameters. 
Initial learning rate was set $\eta = 0.001$ and manually 
updated until the loss value hits plateau. 


#### 2. Color vs Gray dataset

We started from LeNet5 as the baseline model. 
The validation accuracy were 85.28% and 82.47%
 for augmented grayscaled data and augmented RGB data, respectively. 
Though it has a smaller number of parameters, 
the model trained with gray scaled dataset consistently performed better
than the color dataset as it was reported in Sermanet and Lecun(2010).

#### 3. Impact of scale parameter for augmentation

We experimented with three scale paramters, 1, 2, and 4, 
for cnn_base and cnn_deep, smaller models among 
our model candidates. 
The parameter weights_for_scarce was fixed at 0.75.
Both models showed better validation accuracy
 when trained with more examples. 


| Model    | Num. Parameters|Scale parameter(s)|  Num. Examples| Val. accuracy |
| ---------|----------------|------------------|---------------|--------------|
| CNN_base |563,035         | 1                | 34,799        |0.9199 |
| CNN_base | 563,035        | 2                |55,257         |0.9202 |
| CNN_base | 563,035        | 4                | 99,176        |0.9342 |
| CNN_deep |201,331         | 1                |34,799         |0.9100 |
| CNN_deep |201,331         | 2                |55,257         |0.9222 |
| CNN_deep |201,331         | 4                |99,176         |0.9438 |  


#### 4. Architecture selection experiments

Further, we explored various combinations of model architecture. 
For details, refer to [this file](https://github.com/JennyLeeStat/TrafficSigns/blob/master/cnn_models.py)
and corresponding checkpoints. 

After observing our base model underfitted given augmented training dataset, 
we tried tweaking several model parameters to increase the model capacity.
The width of each layer was widened, additional layers were stacked to make model deeper,
and different combination of kernel sizes were tested. 
With such large parameters, the model quickly overfitted even with 
augmented dataset. 
Drop out layers were used in the dense layers 
to prevent overfitting. 

The table below compares the model capacity and validation accuracy
when trained with augmented (s=4, w=0.75) grayscaled training dataset (n=99,176).
By boosting model capacity and number of input images,
all models showed improved validation accuracy compared to 
LeNet, and achieved over 93%, our project goal.


| Model                      |   Num. Parameters| Val. accuracy |
|----------------------------|----------------:|--------------:|
|cnn_small_kernel_deep_wide  | 642,459         | 0.9683|
|cnn_small_kernel_deep_wide2 |1,188,507          | 0.9676|
|cnn_wide                    | 557,555         | 0.9633 |
|cnn_deepx2_wide             | 406,555         | 0.9607|
|cnn_wider                   | 2,802,803	       |0.9601|
|cnn_wider2                  |2,201,179	        |0.9560|
|cnn_small_kernel_wide       |2,197,147	        |    0.9535|
|cnn_best                    |319,387	       |0.9517|
|cnn_wide_deep               |352,155	       |0.9469|
| cnn_deep                   |201,331	        |0.9438|
| cnn_small_kernel_deep      | 310,939         |	0.9426|
|cnn_base                    |563,035	      |0.9342|
|cnn_small_kernel            | 554,587	       |0.9324|
|lenet (base model)         | 84,107	        | 0.8528|



#### 5. Final model architecture: cnn_small_kernel_deep_wide

The best performing model was cnn_small_kernel_deep_wide which
consisted of the following layers:

| Layer | Description      |Maps | Size | Kernel size | Stride | Activation|
|-------|------------------|-----|:-----:|-------------|--------|-----------|
| Out   |  output          |  -  | 43   |  -          |  -     | -         |
| FC5   | Fully connected  |  -  |  84  |  -          |  -     | ReLU      |
| FC4   | Fully connected  |  -  |  256 |  -          |  -     | ReLU      |
| Pool3 |  Maxpooling      | 128  | 4 x 4  |   2 x 2   |   2    |   -       |
|Conv3  | Convolution      | 128  | 8 x 8  |   3 x 3   |   1    |   ReLU    |   
| Pool2 | Maxpooling       | 64  | 8 x 8  |   2 x 2   |   2    |   -       |
| Conv2 | Convolution      | 64  | 16 x 16|   3 x 3   |    1   |   ReLU    |
| Pool1 | Maxpooling       | 32  | 16 x 16|   2 x 2   |    2   |    -      |
| Conv1 | Convolution      | 32  | 32 x 32|   3 x 3   |    1   |   ReLu    |
| In    | Input            | 3   | 32 x 32|   -       |    -   | -         |


The hyperparameters used to train the best model was:

| Hyperparameter | value |
| ---------------|-------|
| Batch size     | 128 |
| Num of epochs  | 20 |
|Learning rate  | 0.001|
|Drop out rate    | 0.5   |
| Initial weights | truncated $Normal(0, 0.1)$ |
| Initial biases | 0 |
 
 


My final model results were:

| Dataset | Accuracy | Precision | Recall |
|---------|----------|-----------|--------|
| Train (not augmented)   |  0.9994    | 0.9994  |  0.9994      |
| Validation |  0.9685 | 0.9712  | 0.9685    |
| Test    |   0.9518      |    0.9532      |    0.9518    |


#### 6. Error analysis



The normalized confusion matrix from test set is visualized below. 
To visualize only errors, correct predictions on the diagonal 
were set to zero. Bright color represents high error rate. 
The final model seems to confuse class 12 (Priority road) 
for class 40 (Roundabout mandatory).
It also confuses class 25(Road work) and 
class 26 (Traffic signals) for class 27 (Pedestrians).
It might make sense because all three is triangular sign 
with black picture in them. 



![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/error_analysis.png)







---

# Test a Model on New Images

#### 1. Five German traffic signs found on the web

Five German traffic signs found on the web were used to test the final model.

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/test_images5.png)


- First image was selected because class 12 (Priority road) had
the highest normalized error rate. 
- The second image could be hard 
for the classifier as the outer circle is not clearly shown. 
- Electronic traffic signs were not included in the 
training, valid, or test dataset. The third image was selected to see 
if the classifier can generalize and correctly predict the true label
for the images it has never seen.
- Class 30 (Beware of ice/snow) was wrongly classified as class 23
(Slippery road) at higher rate. Additionally, 
it could be hard for the classifier because on the selected image
the sign was partially covered with ice and the object of interest 
is not centered. 
- Class 25 was also among the classes that showed higher error rate. 
The non-smooth surface of the sign and characters which were not included in the 
training dataset could make the classification harder. 


#### 2. Top predictions by model
Discuss the model's predictions on these new traffic signs and 
compare the results to predicting on the test set. 
At a minimum, discuss what the predictions were, 
the accuracy on these new predictions, 
and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| True label | Top Predicted |
|------------|-----------|
| Priority road| Priority road |
| Speed limit (30km/h)|  Speed limit (30km/h)|
| Speed limit (120km/h)| Stop                |
| Beware of ice/snow|  Slippery road   |
| Road work|   Speed limit (30km/h)  |


The model predicted only two out of five images correctly, 
results in 40% of accuracy. This results are somewhat disappointing 
considering it showed over 95% accuracy on test set. 
This might indicate that 
our model was too powerful and tuned to German Traffic Signs dataset
so much and might have failed to generalize to real world images. 



#### 3. Top five softmax probability


- The model correctly predicted the true label of the first test image
and very certain. 

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/top5_test_1.png)

- It classified the second image correctly 
and very confidently. Other top choices were all speed limit signs. 

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/top5_test_2.png)

- Though the model confidently selected its top choice, 
the true label was not in the top 5 choices. The training dataset 
our best model was trained on did not included electronic traffic signs.
Somewhat disappointingly, the model failed to generalize to the 
images it has never seen. 

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/top5_test_3.png)

- The model was not certain in its decision. After lowering 
 resolution for preprocessing, frankly, it's hard to recognize even in human eyes. 
 The true label, Beware of ice/snow, was on the third of its top choices. 
As pointed out in our error analysis by confusion matrix, 
the model misclassified Beware of ice/snow as Slippery road. 

 
![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/top5_test_4.png)

 
-  Again, though the model confidently selected its top choice, 
the true label was not in the top 5 choices. 

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/top5_test_5.png)


# Visualizing the Neural Network 

A sample image of road work sign was forward passed and 
activations after each of first three convolutional layers were visualized below.
The number of filters for these conv layers are \[32, 64, 128] in the final model. 
The images in each layer represent filters for that layer.   

The featuremaps from first conv layer shows it captures shape and contrast quite well already. 
Other activations are highly summarized and not easily interpretable. 

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/activation_first_layer.png)

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/activation_second_layer.png)

![Alt text](https://github.com/JennyLeeStat/TrafficSigns/blob/master/images/activation_third_layer.png)



# Reference

\[1] Deep Learning, I. Goodfellow, Y. Begio, and A. Courville (2016)

