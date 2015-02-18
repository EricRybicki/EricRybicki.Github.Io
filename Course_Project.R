training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
names(training)
str(training)
library(randomForest)
library(dplyr)
library(caret)

set.seed(212)
#Remove first seven columns
training <- select(training, -(X:num_window))
#Calculate and remove near zero variables
nsv <- nearZeroVar(training, saveMetrics = T)
training <- training[, !nsv$nzv]
rm(nsv)
#Calculate and remove columns with 90% missing values
nav <- sapply(colnames(training), function(x) if(sum(is.na(training[, x])) > 0.9*nrow(training)){return(T)}else{return(F)})
training <- training[, !nav]
rm(nav)

#Created training and testing sets
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
training.set <- training[inTrain, ]; testing.set <- training[-inTrain, ]
dim(training.set); dim(testing.set)
rm(training)


#Random Forest
modFit.rForest <- randomForest(classe ~. , data=training.set)
modFit.rForest
train.pred <- predict(modFit.rForest, testing.set, type = "class")
confusionMatrix(train.pred, testing.set$classe)


# Assign predictions to testing set using model
test.pred <- predict(modFit.rForest, testing)


# Create Answer files
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(test.pred)
