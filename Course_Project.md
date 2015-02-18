---
title: "Practical Machine Learning - Course Project"
author: "EricRybicki"
date: "18 Feb 2015"
output: html_document
---
## Background
This data set consists of accelerometer data of six participants taken from the belt, forearm, and dumbbell. The participants were asked to perform barbell lifts correctly and incorrectly in five different ways. These exercises were each classified in the data set.

## Prediction Assignment
For this course project in the Practical Machine Learning course on the Data Science Specialization offered by Johns Hopkins University provided by Coursera.org, we are given two data sets and must work to find an approximate model which might predict how the team which compiled the data sets classified the exercise. 

### Data
The [Training set](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and the [Testing set](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) is provided on the course site on [Coursera.org](https://class.coursera.org/predmachlearn-011/human_grading/view/courses/973546/assessments/4/submissions) and the following code is written on the assumption that the data has been manually downloaded into your working directory. 


```r
library(randomForest)
library(dplyr)
library(caret)
set.seed(212)
```

Load data

```r
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
head(names(training), 12)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "kurtosis_roll_belt"
```

#### Data Cleaning
At first glance we notice that several columns have unique value columns which might contribute to over-fitting of data (eg ```user_name```) and troublesome factor data (eg ```cvtd_timestamp```). Thus the first order of business is to remove the first seven columns. Next we want to clean the columns which consist of primarily one unique value. Finally we notice the nontrivial amount of missing data, so we remove the columns with more than ```90%``` missing data. 

```r
#Drop first seven rows
training <- select(training, -(X:num_window))
#Determine and remove near zero variance columns
nsv <- nearZeroVar(training, saveMetrics = T)
training <- training[, !nsv$nzv]
rm(nsv)     #remove helper variable
#Calculate and remove columns with 90% missing values
nav <- sapply(colnames(training), function(x){
                                 if(sum(is.na(training[, x])) > 0.9*nrow(training)){return(T)}else{return(F)}
                                 }
                )
training <- training[, !nav]
rm(nav)     #remove helper variable
```

### Spliting data
It is always good practice to split your training data set into a further training and testing set so that you are able to casually test your trained model on an untapped test set, while leaving the official test set untouched until the model is ready to be applied.

```r
#Split cleaned data into training and testing sets
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
training.set <- training[inTrain, ]     #This alocates 70% of the cleaned data into the new training set
testing.set <- training[-inTrain, ]     #And 30% of the cleaned data into our casual testing set
rm(training)    #Remove original training data from memory
```

### Random Forests
The data is now ready to construct a model which might predict the exercise classification in the data set. I have chosen to begin with a random forest model to predict ```classe``` against all other predictors. 

```r
#Random Forest
modFit.rForest <- randomForest(classe ~. , data=training.set)
modFit.rForest
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training.set) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.58%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3899    4    0    1    2 0.001792115
## B   15 2638    5    0    0 0.007524454
## C    0   14 2380    2    0 0.006677796
## D    0    0   24 2226    2 0.011545293
## E    0    0    3    8 2514 0.004356436
```
From the random forest model we expect an ```out of sample error rate``` to be ```0.58%```.
We will use this random forest model to predict the exercise classification on the test set.

```r
train.pred <- predict(modFit.rForest, testing.set, type = "class")
confusionMatrix(train.pred, testing.set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    7    0    0    0
##          B    1 1127    5    0    0
##          C    0    5 1021   14    0
##          D    0    0    0  950    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9944          
##                  95% CI : (0.9921, 0.9961)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9929          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9895   0.9951   0.9855   0.9991
## Specificity            0.9983   0.9987   0.9961   0.9998   1.0000
## Pos Pred Value         0.9958   0.9947   0.9817   0.9989   1.0000
## Neg Pred Value         0.9998   0.9975   0.9990   0.9972   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1915   0.1735   0.1614   0.1837
## Detection Prevalence   0.2855   0.1925   0.1767   0.1616   0.1837
## Balanced Accuracy      0.9989   0.9941   0.9956   0.9926   0.9995
```

### Results and Conclusion
Finally we apply our model to our given testing set to determine if our model accurately predicts the classification of exercises. 

```r
test.pred <- predict(modFit.rForest, testing)
```

We generate answers to our assignment with the following code given found on the project instructions page on [coursera.org](https://class.coursera.org/predmachlearn-011/assignment/view?assignment_id=5). The happy surprise of all the answer being correct indicating a 100% prediction rate suggests that this model is sufficient for this assignment and that no further exploratory modeling is necessary. 

### Submission code

```r
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
pml_write_files(test.pred)
```
