Aaron Andrews
=============

Project 4 / Scene Recognition with Bag of Words
-----------------------------------------------

The purpose of this assignment is to explore certain methods of scene
recognition. Or more specifically, classifying images by mapping them to
categories (e.g 'Bedroom' or 'Forest') based on ground truth training
images already tied to said categories. The basic process for this is to
featurize our test images and trainging images in some fashion, then for
each test feature, find the closest training features and then choose
the category for that training data as the classification for the image.
The most basic methods I implemented for this are tiny images and k
nearest neighbors classification, but I yielded better accuracy by using
bags of local SIFT features and linear support vector machine
classification.

### Tiny Images + k Nearest Neighbors

The tiny images method is rather simple, actually. All I do is take the
image, shrink it to 16 x 16 (ignoring aspect ratio), and then flatten
out to a 256 size vector, and BAM, thats our 'feature'.

Tiny Images
-----------

    image_feats = [];
    for i = 1:size(image_paths, 1)
        filename = image_paths(i);
        filename = char(filename);
        
        B = imread(filename);
        B = imresize(B, [16 16]);
        
        vector = reshape(B, 1, []);
        vector = normr(vector);
        image_feats = [image_feats; vector];
        
    end

The end result result is an N x d matrix representing our image
features, where N is the number of images, and d is simply the
dimensionality of the image's feature, which, in this case, should be
uniformly 256 for each image.

After getting the feature matrices for both the training and test images
using this method, we now handle classification of the test images using
a simple k Nearest Neighbors algorithm. The reason we're doing kNN
instead of simply finding the absolute closest match for each image
feature is to dampen the influence of noise in our data, which can give
us false positives and greatly lower our accuracy. Instead I looked at k
of the closest matches, and within that list of test features, I took a
vote for each category represented in the set, and the category with the
most votes was chosen as the classification for each image. I found
decent accuracies with k=20 features.

With these two methods combined I got a mean accuracy of 21% for all 15
categories. You can see a breakdown of that [here](tiny_kNN.html), but,
yeah, that's not all that great. To get this accuracy up, I realized
that I would need something a bit more complicated for our features than
just literal shrunk down images.

### Bag of Sifts + k Nearest Neighbors

A bag of sifts is basically a feature histogram that looks at the local
SIFT features of each image, and puts them each in a bin representing
the visual 'word' that describes them. This 'word' is a feature from a
set of features known as a vocabulary. This vocabulary is formed by
sampling a bunch of local SIFT features from the training image set.
This process can take a real long time, so I made sure to save that
vocab for future runs.

The feature histogram, or bag of sifts, is then used as the feature
vector for each image. Using this method with Nearest Neighbors
classification, we found a mean accuracy of around 51.5% which is way
better than tiny images. By building a visual vocabulary of 'words' from
our training image set and then comparing the features of our test
images to those words, we were able to be a lot more accurate. You can
see a breakdown [here](http://htmlpreview.github.io/?https://github.com/brohand/SIFT-and-SVM-Image-Classifier/blob/master/html/BagOfSifts_kNN.html)

### Bag of Sifts + Support Vector Machine

The thing with k Nearest Neighbors and our bag of sifts is that a lot of
the words in our vocabulary can be rather uninformative, between our
training images, there's bound to be a lot of identical 'words' like
smooth patches and step edges between them. But the sheer frequency of
these words can heavily influence our kNN algorithm. That's why, in the
effort of increasing accuracy, I opted for using a Support Vector
Machine to classify the images.

The SVM is basically a learning model that we generate for each category
in our category set. These models basically tell us wether something is
a kitchen or not, or a forest or not by training each of these models on
the training image set. We then evaluate these models with each test
image, and choose the category with the most confidence as our category
for the image.

With this method we got a better accuracy of around 60% with the
lambda=0.000003 for our svm training function. You can see more
[here](http://htmlpreview.github.io/?https://github.com/brohand/SIFT-and-SVM-Image-Classifier/blob/master/html/BagOfSifts_SVM.html)

### Tiny Images + Support Vector Machine

[here](http://htmlpreview.github.io/?https://github.com/brohand/SIFT-and-SVM-Image-Classifier/blob/master/html/tiny_SVM.html)
