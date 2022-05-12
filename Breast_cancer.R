install.packages("PerformanceAnalytics")
library("PerformanceAnalytics")
library(dplyr)
library(ggplot2)
library(gridExtra)
library(arm)
#library(car)
library(psych)
library(caTools)
library(caret)
#library(keras)
library(mice)
library(VIM)
library(rpart)
library(rattle)
library(randomForest)
#load dataset and add column names
canc <- read.csv(file.choose(), header = F ,col.names = c("class","age", "menopause", "tumor.size", "inv.nodes", 
      "node.caps", "deg.malig", "breast", "breast.quad","irradiat"),colClasses = "factor", na.strings = c("?"))
names(canc)
str(canc)
head(canc)
summary(canc)



#explore before cleaning
####Data Exploration######
# will add more graphs
p1 = ggplot(canc, aes(x=inv.nodes, fill=class)) + geom_bar(position='dodge') + labs(title='Histogram of Inv Nodes Grouped by Class',x='Inv Nodes',y='Count')
p2 = ggplot(canc, aes(x=menopause, fill=class)) + geom_bar(position='dodge') + labs(title='Histogram of Menopause Grouped by Class',x='Menopause',y='Count')
p3 = ggplot(canc, aes(x=irradiat, fill=class)) + geom_bar(position='dodge') + labs(title='Histogram of Irradiat Grouped by Class',x='Irradiat',y='Count')
p4 = ggplot(canc, aes(x=age, fill=class)) + geom_bar(position='dodge') + labs(title='Histogram of Age Grouped by Class',x='Age',y='Count')
p5 = ggplot(canc, aes(x=breast.quad, fill=class)) + geom_bar(position='dodge') + labs(title='Histogram of Breast Quandrant Grouped by Class',x='Breast Quandrant',y='Count')
p6 = ggplot(canc, aes(x=tumor.size, fill=class)) + geom_bar(position='dodge') + labs(title='Histogram of Tumor size Grouped by Class',x=' Tumor size',y='Count')
p7 = ggplot(canc, aes(x= node.caps, fill=class)) + geom_bar(position='dodge') + labs(title='Histogram of node.caps Grouped by Class',x='Node.caps',y='Count')
p8 = ggplot(canc, aes(x=breast, fill=class)) + geom_bar(position='dodge') + labs(title='Histogram of Breast Grouped by Class',x='Breast',y='Count')
p9 = ggplot(canc, aes(x=deg.malig, fill=class)) + geom_bar(position='dodge') + labs(title='Histogram of Deg.malig Grouped by Class',x='Degree of Malignancy',y='Count')
grid.arrange(p1,p2,p3,p4,p5,p6,p7,p8,p9)


#click on zoom if nothing displays in the plot

#no.of.splits <- apply(x=canc[-1], MARGIN = 2, FUN = function(col))


#missing value analysis
colSums(is.na(canc))
canc[!complete.cases(canc),]
md.pattern(canc)
#marginplot(canc[, c("deg.malig", "node.caps")])

#####Calculate Proportion of missing data
prop <- function(x) {sum(is.na(x))/ length(x)*100}
apply(canc,2,prop)

#impute missing data
impute <- mice(canc, m=3, seed = 123)
print(impute)
impute$imp$node.caps
impute$imp$breast.quad
canc[c(146,164,165,184,185,234,264,265),]
canc[c(206:210),]

#completed dataset
cancer_clean <- complete(impute, 3)
colSums(is.na(cancer_clean))
canc[!complete.cases(cancer_clean),]
md.pattern(cancer_clean)
#correlation matrix
#chart.Correlation(canc, histogram=TRUE, pch=19)
#pairs.panels(canc)

#splitting the data
set.seed(777)
train.index <- sample(1:nrow(cancer_clean), 0.7* nrow(cancer_clean))
print(sort(train.index))
cancer.train <- cancer_clean[train.index,]
dim(cancer.train)
cancer.test <- cancer_clean[-train.index,]
dim(cancer.test)
#check proportion of train and test data
prop.table(table(cancer.train$class))
prop.table(table(cancer.test$class))
#prop.table(table(cancer_clean$class))

#building the model Decision tree
set.seed(777)
cancer.tree <- rpart(class ~., data = cancer.train, method = "class")
print(cancer.tree)
summary(cancer.tree)
fancyRpartPlot(cancer.tree, caption = NULL)


#Random Forest Model
set.seed(777)
cancer.rf.model <- randomForest(class ~.,data = cancer.train, importance = TRUE, ntree = 100, mtry = 2)
print(cancer.rf.model)
importance(cancer.rf.model)
varImpPlot(cancer.rf.model)
#Explain these mean Decrease in the report


###########Model evaluation##################

#predicting the model
cancer.predictions <- predict(cancer.tree, cancer.test, type = "class")
head(cancer.predictions)
#comparison table
cancer.comparison <- cancer.test
cancer.comparison$predictions <- cancer.predictions
cancer.comparison[,c("class", "predictions")]

#View misclassified rows
disagreement.index <- cancer.comparison$class != cancer.comparison$predictions
cancer.comparison[disagreement.index,]


#Decision tree confusion Matrix
tree.confusion <- table(cancer.predictions, cancer.test$class)
print(tree.confusion)


#Overall recall
overall.recall<-(tree.recall.A+tree.recall.B)/2
print(overall.recall)

tree.f1 <- 2 * overall.precision * overall.recall / (overall.precision + overall.recall)
print(tree.f1)

confusionMatrix(tree.confusion, mode = "prec_recall")

###########Forest EValuation##########

#Predict with test data

preds.rf.cancer <- predict(cancer.rf.model, cancer.test)
table(preds.rf.cancer, cancer.test$class)

## calculate the confusion matrix
cancer.rf.confusion <- table(preds.rf.cancer, cancer.test$class)
print(cancer.rf.confusion)
## accuracy
cancer.rf.accuracy <- sum(diag(cancer.rf.confusion)) / sum(cancer.rf.confusion)
print(cancer.rf.accuracy)


confusionMatrix(cancer.rf.confusion, mode = "prec_recall")


########parameter Tunning#########
set.seed(777)
tree.params <- rpart.control(minsplit=3, minbucket=round(5 / 3), maxdepth=15, cp=0.01169591)

## Fit decision model to training set
## Use parameters from above and Gini index for splitting
plotcp(cancer.tree)
cancer.tree <- rpart(class ~ ., data = cancer.train, 
                     control=tree.params, parms=list(split="gini"))

cancer.predictions <- predict(cancer.tree, cancer.test, type = "class")
head(cancer.predictions)
cancer.comparison <- cancer.test
cancer.comparison$predictions <- cancer.predictions
cancer.comparison[,c("class", "predictions")]

tree.confusion <- table(cancer.predictions, cancer.test$class)
print(tree.confusion)
confusionMatrix(tree.confusion, mode = "prec_recall")


##########Forest Tunnung/boosting#########

set.seed(1232)

#=================================================================
# Train Model -Random Forest
#=================================================================

# Set up caret to perform 10-fold cross validation repeated 3 
# times and to use a grid search for optimal model hyperparamter
# values.

train.control <- trainControl(method = "repeatedcv",
                              number = 100,
                              repeats = 3,
                              search = "grid",
                              allowParallel = T)

# Leverage a grid search of hyperparameters for randomForest. See 
# the following presentation for more information:

tune.grid <- expand.grid(mtry = c(3:6))

View(tune.grid)

#Let's run it parallely
install.packages("doParallel")

library(doParallel)

cl <- makePSOCKcluster(8)
registerDoParallel(cl)

# Train the randomforest model using 10-fold CV repeated 3 times 
# and a hyperparameter grid search to train the optimal model.
forest.boost <- train(class ~ ., 
                  data = cancer.train,
                  method = "rf",
                  tuneGrid = tune.grid,
                  trControl = train.control)

print(forest.boost)

stopCluster(cl)

# insert serial backend, otherwise error in repetetive tasks
registerDoSEQ()

# Make predictions on the test set using a randomForest model 
# trained on all  rows of the training set using the 
# found optimal hyperparameter values.
preds.rf.boost <- predict(forest.boost, cancer.test)

table(preds.rf.boost,cancer.test$class)
# Use caret's confusionMatrix() function to estimate the 
# effectiveness of this model on unseen, new data.
Confusion.rf.boost<-confusionMatrix(preds.rf.boost, cancer.test$class, mode = "prec_recall")
Confusion.rf.boost

#####Gradient boosting ########
install.packages("gbm")
tune.grid.b <- expand.grid(n.trees = c(100,500), interaction.depth=c(1:3), shrinkage=c(0.01,0.1), n.minobsinnode=c(20))

View(tune.grid.b)

#Register parallel cores
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

forest.boost <- train(class ~ ., 
                      data = cancer.train,
                      method = "gbm",
                      tuneGrid = tune.grid.b,
                      trControl = train.control)

forest.boost
best_mtry <- forest.boost$bestTune$interaction.depth
stopCluster(cl)

# insert serial backend, otherwise error in repetetive tasks
registerDoSEQ()

preds.rf.boost <- predict(forest.boost, cancer.test)

table(preds.rf.boost,cancer.test$class)
# Use caret's confusionMatrix() function to estimate the 
# effectiveness of this model on unseen, new data.
Confusion.rf.boost<-confusionMatrix(preds.rf.boost, cancer.test$class, mode = "prec_recall")
Confusion.rf.boost







#Clear environment
#rm(list=ls())


 