# Accuracy

In classification problems is the **number of correct predictions** made by the model divided by the **total number of predictions.**  
For example, if the X_test set was 100 images and our model **correctly** predicted 80 images, then we have **80/100**.  
**0.8** or **80% accuracy.**  
Accuracy is useful when target classes are well balanced.  
In our example, we would have roughly the same amount of cat images as we have dog images.  
Accuracy is **not** a good choice with **unbalanced** classes!  
Imagine we had 99 images of dogs and 1 image of a cat.  
If our model was simply a line that always predicted **dog** we would get 99% accuracy!  

# Recall

Ability of a model to find all the relevant cases within a dataset.  
The precise definition of recall is the number of true positives divided by the number of true positives plus the number of false negatives.  

# Precision

Ability of a classification model to identify only the relevant data points.  
Precision is defined as the number of true positives divided by the number of true positives plus the number of false positives.

# Recall X Precision

Often you have a trade-off between Recall and Precision.  
While recall instances in a dataset, precision expresses the proportion of the data points our model says was relevant actually were relevant.

# F1-Score

In cases where we want to find and optimal blend of precision and recall we can combine the two metrics using what is called the F1 score.  
The F1 score is the harmonic mean of precision and recall taking both metrics into account in the folllowing equation:
```
F1 = 2 * (precision * recall)
         --------------------
         (precision + recall)
```
We use the harmonic mean instead of a simple average because it punishes extreme values.  
A classifier with a precision of 1.0 and a recall of 0.0 has a simple average of 0.5 but an F1 score of 0 (zero).

## Confusion Matrix

We can also view all out correctly classified versus incorrectly classified images in the form of a confusion matrix.  
The main point to remember with the confusion matrix and the various calculated metrics is that they are all fundamentally ways of comparing the predicted values versus the true values.  
What constitutes "good" metrics, will really depend of the specific situation!  
We can use a consfusion matrix to evaluate our model.  
For example, imagine testing for disease.  
[Confusion Matrix 1](./metrics%201.png)
[Confusion Matrix 2](./metrics%202.png)  
For the examples:  
- Accuracy: Overall, how often is it correct? (TP+TN)/total = 150/165 = 0.91
- Misclassification Rate (Error Rate) = 1 - Accuracy = 0.9