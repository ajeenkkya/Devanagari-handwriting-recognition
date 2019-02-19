# Devanagari Character Recognition

## Introduction
This is a project of Devanagari Character recognition, Devanagari is a script on which languages like Hindi, Sanskrit, Awadhi etc. are based. The dataset contains 72000 images which contains 2000 images of each letter, there are 36 letters in total. The dataset is a csv file which contains the gray values of the image pixel wise, as a single image is of 32x32 so there are 1024 pixels and thus 0-1023 graylevels of pixels stored in each column. Also, 1 column of the labels of each image. So, that makes the dataset of 1025 columns and 72000 rows. You can get the Data folder of images from https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset#
Also the CSV file which is the most vital part of the project can’t be uploaded to the Github because of its size so you can get it here:
https://drive.google.com/open?id=1soPwPRBhdpZsswSwa9SrRnWP8BK24xWJ

## The Model
The network here used is Sequential and as shown in the figure below:
![img_20190220_010917](https://user-images.githubusercontent.com/35074988/53045838-a942dd80-34b4-11e9-868c-e25831491f72.jpg)
In the above figure the first matrix is of 32x32x1 which is just the image, here 1 signifies that the image is black and white image, if it would have been a colour image then it would have been 32x32x3. Then we convolute the image with a filter of 5x5 and we apply 32 filters, which makes it a matrix of 28xx28x32. Then we apply MaxPooling with filter 2x2 and stride of (2,2) i.e. after considering the maximum of the first 2x2 filter, the filter will move by 2 element to the right; and after completing the row it will move 2 element downwards and continue the process. Also, the padding is set to ‘same’ which basically means that it will add padding if the elements aren’t even(https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t). Similarly we perform convolution and MaxPooling again we perform flattening of the nodes i.e. converting it to a vector and then applying dense function which distributes the nodes according to the classes.
You can use my pre-trained model which is saved as devanagari.h5

## Problems I faced
The original dataset had string value as labels but because i wanted to use to_categorical function so i had to change the labels to an integer value of simply 1-36 depending on the class it belongs.
Also, if the images that i gave as input were not cropped till the edges of the character then it classified incorrectly and if I gave the image of the letters written in a notebook even then it classified incorrectly. One possible reason is that the images on which the model was trained on was black background and white font so it struggled with different conditions. As soon I wrote in a black background and white text the prediction was perfect(well near about perfect as the accuracy was nearly 98%).
