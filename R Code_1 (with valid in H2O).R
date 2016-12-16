
setwd("C:/Users/Amol Jadhav/Desktop")

getwd()

# 1.Load packages ###################################################
install.packages(FSelector)
library(FSelector) # need JRE!
library(caret)
library(randomForest)
library(kernlab)
library(e1071)

# 2.Load data #######################################################
## set up urls for datasets
### common data
url_variables <- "./UCI HAR Dataset/features.txt"
url_activity_names <- "./UCI HAR Dataset/activity_labels.txt"
### training data
url_train_data <- "./UCI HAR Dataset/train/X_train.txt"
url_train_activity <- "./UCI HAR Dataset/train/y_train.txt"
url_train_subjects <- "./UCI HAR Dataset/train/subject_train.txt"
### test data
url_test_data <- "./UCI HAR Dataset/test/X_test.txt"
url_test_activity <- "./UCI HAR Dataset/test/y_test.txt"
url_test_subjects <- "./UCI HAR Dataset/test/subject_test.txt"

## load common data
activity_names <- read.table(url_activity_names, stringsAsFactors=F)
var_names <- read.table(url_variables, stringsAsFactors=F)
## load training data
train_data <- read.table(url_train_data)
train_activity <- read.table(url_train_activity)

head(train_data)
head(train_activity)

## load test data
test_data <- read.table(url_test_data)
test_activity <- read.table(url_test_activity)
## Merges data ######################################################
## correct variable names
## editNames function for the substitution for variable names
editNames <- function(x) {
  y <- var_names[x,2]
  y <- sub("BodyBody", "Body", y) #subs duplicate names
  y <- gsub("-", "", y) # global subs for dash
  y <- gsub(",", "_", y) # global subs for comma
  y <- sub("\\()", "", y) # subs for ()
  y <- gsub("\\)", "", y) # global subs for
  y <- sub("\\(", "_", y) # subs for (
  y <- paste0("v",var_names[x,1], "_",y) #add number, prevent duplicates
  return(y)
}
## edit names
new_names <- sapply(1:nrow(var_names), editNames)

## work with training data
names(train_data)<-new_names
train_data <- cbind(train_activity[,1], train_data)
names(train_data)[1]<-"Activity"

## work with test data
names(test_data) <- new_names
test_data <- cbind(test_activity[,1], test_data)
names(test_data)[1]<-"Activity"
activity_names[2,2] <- substr(activity_names[2,2], 1, 10) #cut long names
activity_names[3,2] <- substr(activity_names[3,2], 1, 12)
train_data <- transform(train_data, Activity=factor(Activity))
test_data <- transform(test_data, Activity=factor(Activity))
levels(train_data[,1])<-activity_names[,2]
levels(test_data[,1])<-activity_names[,2]

# 3. Data exploration ###############################################
## check range
rng <- sapply(new_names, function(x){range(train_data[,x])})                               
min(rng)
max(rng)
## check skewness
SkewValues <- apply(train_data[,-1], 2, skewness)
head(SkewValues[order(abs(SkewValues),decreasing = T)],3)
## activity distribution
summary(train_data$Activity)

summary(train_data)
str(train_activity)


# 4. Full models ####################################################
## Random Forest
fitControl <- trainControl(method="cv", number=5)
set.seed(12345)
tstart <- Sys.time()
forest_full <- train(Activity~., data=train_data,
                     method="rf", do.trace=10, ntree=400,
                     importance =TRUE,
                     trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)
summary(forest_full)
plot(forest_full)
varImp(forest_full)


## predict and control Accuracy

prediction <- predict(forest_full, newdata = test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)

## SVM, full set
fitControl <- trainControl(method="cv", number=7)
tstart <- Sys.time()
svm_full <- train(Activity~., data=train_data,
                  method="svmRadial",
                  trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)

summary(svm_full)
plot(svm_full)

## predict and control Accuracy
prediction <- predict(svm_full, newdata=test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)


# 5. Model with important variables #################################
plot(varImp(forest_full),20, scales=list(cex=1.1))

## % variable extraction ######################
imp <- varImp(forest_full)[[1]]
imp_vars <- rownames(imp)[order(imp$Overall, decreasing=TRUE)]  ## Dint work on 01/07/16
vars <- imp_vars[1:490] # % features

summary(imp)

## model
fitControl <- trainControl(method="cv", number=5)
tstart <- Sys.time()
svm_imp <- train(Activity~., data=train_data[,c("Activity", vars)],
                 method="svmRadial", trControl = fitControl)

tend <- Sys.time()
print(tend-tstart)
prediction <- predict(svm_imp, newdata=test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)

# 6. Information gain ###############################################
## calculate ratio
inf_g <- information.gain(Activity~., train_data)
inf_gain <- cbind.data.frame(new_names, inf_g, stringsAsFactors=F)
names(inf_gain) <- c("vars", "ratio")
row.names(inf_gain) <- NULL
## arrange by ratio descending and plot top-20 variables
inf_gain <- inf_gain[order(inf_gain$ratio, decreasing=TRUE),]
dotplot(factor(vars, levels=rev(inf_gain[1:20,1])) ~ ratio,
        data=inf_gain[1:20,],
        scales=list(cex=1.1))
inf_gain[10,1]
## [1] "tBodyAccmadX"
plot(train_data[,inf_gain[10,1]], ylab=inf_gain[10,1],
     pch=20, col=train_data[,1], main="IGR = 0.87")
legend("topright", pch=20, col=activity_names[,1],
       legend=activity_names[,2], cex=0.8)
inf_gain[551,1]
## [1] "tBodyAccJerkMagarCoeff4"
plot(train_data[,inf_gain[551,1]], ylab=inf_gain[551,1],
     pch=20, col=train_data[,1], main="IGR = 0.03")
legend("topright", pch=20, col=activity_names[,1],
       legend=activity_names[,2], cex=0.8)

## select variables (igr cutoff) ################
vars <- inf_gain$vars[1:547]
## SVM with best igr variables ##################
## for parallel processing
# library(doMC) # don't use for Windows
# registerDoMC(cores=3) # don't use for Windows
fitControl <- trainControl(method="cv", number=7, allowParallel = TRUE)
tstart <- Sys.time()
svm_igr <- train(Activity~., data=train_data[,c("Activity", vars)],
                 method="svmRadial",
                 trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)
prediction <- predict(svm_igr, newdata=test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)

## Random Forest ################################
vars <- inf_gain$vars[1:526] # Accuracy = 0.9243
fitControl <- trainControl(method="cv", number=5)
set.seed(123)
tstart <- Sys.time()
forest_igr <- train(Activity~., data=train_data[,c("Activity", vars)],
                    method="rf", do.trace=10, ntree=400,
                    trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)

## predict and control Accuracy
prediction <- predict(forest_igr, newdata=test_data)
cm <- confusionMatrix(prediction, test_data$Activity)
print(cm)

## PCA ##############################################################
pca_mod <- preProcess(train_data[,-1],
                      method="pca",
                      thresh = 0.95)
summary(pca_mod)
print(pca_mod)
plot(pca_mod)
biplot(pca_mod, col= c("gray", "black"))

# getting variance 


pca_train_data <- predict(pca_mod, newdata=train_data[,-1])
dim(pca_train_data)
# [1] 7352 102
#Adding back activity so as to run RF, SVM
pca_train_data$Activity <- train_data$Activity
pca_test_data <- predict(pca_mod, newdata=test_data[,-1])
pca_test_data$Activity <- test_data$Activity

library(ggplot2)
par(mar=rep(2,4))
summary(pca_train_data)
plot(pca_train_data, col= c("gray", "black")) ## Gave error figure margins too large

## Tried another round of PCA to get figure 02/07/2016 (Here)
har.pca <- prcomp(train_data[,-1], center=TRUE, scale. = TRUE)
print(har.pca)
plot(har.pca, type ="l")
plot(har.pca)
biplot(har.pca, col= c("gray", "black"))  ## prints components
summary(har.pca)
har.pca_train_data <- predict(har.pca, newdata=train_data[,-1] )


## Awesome variable factor map and scree plot
install.packages("factoextra")
library("factoextra")

eig.val <- get_eigenvalue(har.pca)
head(eig.val)
fviz_screeplot(har.pca, ncp=30)
fviz_pca_var(har.pca, col.var="contrib") + scale_color_gradient2(low="white", mid="blue", high="red", midpoint=55) + theme_minimal()

# Repeating for test 
har.pca2 <- prcomp(test_data[,-1], center=TRUE, scale. = TRUE)
print(har.pca2)
plot(har.pca2, type ="l")
plot(har.pca2)
biplot(har.pca2, col= c("gray", "black"))  ## prints components
summary(har.pca2)


### Plotting individual PCA lungs 
library(ggfortify)
har.df <- train_data[,-1]
autoplot(prcomp(har.df), data = train_data, col = '#24239D') 

str(har.df)
har2.df <- test_data[,-1]
autoplot(prcomp(har2.df), data = test_data, color = 'red') 

## (To here)



## RF with pca data #######################################

fitControl <- trainControl(method="cv", number=5)
set.seed(12345)
tstart <- Sys.time()
forest_pca <- train(Activity~., data=pca_train_data,
                    method="rf", do.trace=10, ntree=400, importance = TRUE,
                    trControl = fitControl)
tend <- Sys.time()
print(tend-tstart)
summary(forest_pca)
print(forest_pca)
plot(forest_pca)


## predict and control Accuracy
prediction <- predict(forest_pca, newdata=pca_test_data)
cm6 <- confusionMatrix(prediction, test_data$Activity)
print(cm6)
# Accuracy : 0.8734

## SVM with pca data ######################################
fitControl <- trainControl(method="cv", number=7)
set.seed(12345)
tstart <- Sys.time()
svm_pca <- train(Activity ~ ., data=pca_train_data,
                 method="svmRadial",
                 trControl = fitControl)
print(svm_pca)
plot(svm_pca)

tend <- Sys.time()
print(tend-tstart)
## predict and control Accuracy
prediction <- predict(svm_pca, newdata=pca_test_data)
cm7 <- confusionMatrix(prediction, test_data$Activity)
print(cm7)

# Accuracy : 0.9386


## Deeep learning ######################################


library(h2o)

## start a local h2o cluster
localH2O = h2o.init(max_mem_size = '6g', # use 6GB of RAM of *GB available
                    nthreads = -1) # use all CPUs (8 on my personal computer :3)

## MNIST data as H2O
# not used pca_train_data[,1] = as.factor(mnist_train[,1]) # convert digit labels to factor for classification
train_h2o = as.h2o(train_data)

test_h2o = as.h2o(test_data)


### 07/06/2016 ..tried train,validation test
df1<- train_h2o
df2 <- test_h2o


splits <- h2o.splitFrame(df1, c(0.7, 0.2), seed=12345)
train <- h2o.assign(splits[[1]], "train.hex")
valid <- h2o.assign(splits[[2]], "valid.hex")
test <- h2o.assign(splits[[3]], "test.hex")

## set timer
s <- proc.time()

## train model
model = h2o.deeplearning(x = 2:98,  # column numbers for predictors
                         y = 1,   # column number for label
                         training_frame = train_h2o, # data in H2O format
                         activation = "RectifierWithDropout", # algorithm
                         input_dropout_ratio = 0.2, # % of inputs dropout
                         hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                         balance_classes = TRUE, 
                         hidden = c(100,100), # two layers of 100 nodes
                         momentum_stable = 0.99,
                         nesterov_accelerated_gradient = T, # use it for speed
                         epochs = 15) # no. of epochs

# with Tan Hyperbolic activation

model2 = h2o.deeplearning(x = 2:562,  # column numbers for predictors ## ** changed on 07/06/2016 to 2:562 from 2:98
                            y = 1,   # column number for label
                            training_frame = train, # data in H2O format
                            validation_frame = valid,
                            activation = "Tanh", # algorithm
                            input_dropout_ratio = 0.2, # % of inputs dropout
                            hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                            balance_classes = TRUE, 
                            hidden = c(100,100), # two layers of 100 nodes
                            momentum_stable = 0.99,
                            nesterov_accelerated_gradient = T, # use it for speed
                            variable_importance =T,
                            epochs = 15) # no. of epochs

# with recitifed linear unit activation
model3 = h2o.deeplearning(x = 2:562,  # column numbers for predictors  ## ** changed on 07/06/2016 to 2:562 from 2:98
                          y = 1,   # column number for label
                          training_frame = train, # data in H2O format
                          validation_frame= valid,
                          activation = "Rectifier", # algorithm
                          input_dropout_ratio = 0.2, # % of inputs dropout
                          hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                          balance_classes = TRUE, 
                          hidden = c(100,100), # two layers of 100 nodes
                          momentum_stable = 0.99,
                          nesterov_accelerated_gradient = T, # use it for speed
                          variable_importance=T,
                          epochs = 15) # no. of epochs




## print confusion matrix
h2o.confusionMatrix(model)

h2o.confusionMatrix(model2)
summary(model2)
h2o.confusionMatrix(model3)
summary(model3)


plot(model2)
h2o.hit_ratio_table(model2, valid=T)[1,2]
model2.fit = h2o.predict(object= model2, newdata= test_h2o)
summary(model2.fit)



plot(model3)
h2o.hit_ratio_table(model3, valid=T)[1,2]
model3.fit = h2o.predict(object= model3, newdata= test_h2o)
summary(model3)
summary(model3.fit)


## print time elapsed
s - proc.time()

## classify test set
h2o_y_test <- h2o.predict(model, test_h2o)

h2o.confusionMatrix(h2o_y_test)

# other way of looking at performance
h2o.performance(model2, test_h2o=test)


# === Not done
## convert H2O format into data frame and  save as csv
df_y_test = as.data.frame(h2o_y_test)
df_y_test = data.frame(ActivityId = seq(1,length(df_y_test$predict)), Label = df_y_test$predict)
write.csv(df_y_test, file = "HAR_submission.csv", row.names=F)
#===

##Gradient Boosting in H2o 


GBmodel1 = h2o.gbm(x = 2:562,  # column numbers for predictors  ## * Changes x= 2:562 on 07/06/2016 from 2:98
                          y = 1,   # column number for label
                          training_frame = train, # data in H2O format
                          validation_frame = valid,
                          ntrees = 30,
                          max_depth = 5,
                          min_rows = 10,
                          balance_classes = TRUE,
                          score_each_iteration = TRUE
                          )

summary(GBmodel1)
h2o.confusionMatrix(GBmodel1)
plot(GBmodel1)
h2o.hit_ratio_table(GBmodel1, valid=T)[1,2]  ## Overall accuracy
#GBmodel1@model
GBmodel1.fit = h2o.predict(object= GBmodel1, newdata= test_h2o)
summary(GBmodel1.fit)
#all_params = lapply(GBmodel1@model, function(x) {x@model$params})

# Variable importance
gbm.VI <- GBmodel1@model$varimp
print(gbm.VI)

##### Naive Bayes
NBmodel1 = h2o.naiveBayes(x = 2:562,  # column numbers for predictors
                          y = 1,   # column number for label
                          training_frame = train_h2o, # data in H2O format,
                          laplace=3)

h2o.confusionMatrix(NBmodel1)
plot(NBmodel1)
NBmodel1.fit = h2o.predict(object= NBmodel1, newdata= test_h2o)
summary(NBmodel1.fit)
print(NBmodel1.fit)

## shut down virutal H2O cluster
h2o.shutdown(prompt = F)
