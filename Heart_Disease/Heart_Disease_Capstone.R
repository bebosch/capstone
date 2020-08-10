#############################################################################################################
# Create train and test set for machine learning algorithm with 7 variables and 297 observations
#############################################################################################################

# install all required libraries (note: this process could take a couple of minutes)
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# load all required libraries
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(knitr)
library(rpart)
library(rpart.plot)

# set digit places to 5
options(digits = 5)

# download metadata
heart <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",header=FALSE,sep=",",na.strings = '?')

# add column names
names(heart) <- c("age", "sex", "chest_pain", "blood_pressure", "cholestoral","blood_sugar", 
                  "electrocardiography", "max_heart_rate", "exercise_angina", "ST_depression", 
                  "slope_peak", "major_vessels", "defect", "disease")

# only attempt to distinguish between presence (values 1,2,3,4) from absence (value 0)
heart <- heart %>% 
  mutate(disease = ifelse(disease > 0, 1, 0))

# remove NAs
heart <- heart %>% 
  na.omit()

# convert selected numeric columns into factor columns
heart <- heart %>% 
  mutate(sex = as.factor(sex)) %>%
  mutate(chest_pain = as.factor(chest_pain)) %>%
  mutate(blood_sugar = as.factor(blood_sugar)) %>%
  mutate(electrocardiography = as.factor(electrocardiography)) %>%
  mutate(exercise_angina = as.factor(exercise_angina)) %>%
  mutate(slope_peak = as.factor(slope_peak)) %>%
  mutate(major_vessels = as.factor(major_vessels)) %>%
  mutate(defect = as.factor(defect)) %>%
  mutate(disease = as.factor(disease))

# only keep variable columns that vary between presence and absence
variables <- c(2, 3, 8, 9, 11, 12, 13, 14)
heart <- heart[, variables]

# create train and test set (20% of data)
set.seed(20, sample.kind="Rounding")
test_index <- createDataPartition(heart$disease, times = 1, p = 0.2, list = FALSE)
train_set <- heart[-test_index,]
test_set <- heart[test_index,]

#############################################################################################################
# To compare different algorithms, we need an accuracy function in order to quantify what it means to do well
#############################################################################################################

# accuracy function
accuracy <- function(predicted_num, true_num){
  confusionMatrix(predicted_num, true_num)$overall[["Accuracy"]]
}

#############################################################################################################
# Performing logistic regression as a baseline approach for further machine learning algorithms
#############################################################################################################

# compute fit of logistic regression algorithm
set.seed(20, sample.kind="Rounding")
glm_fit <- train(disease ~ ., data = train_set, method = "glm", family = "binomial")

# calculate predicted presence of heart disease (with glm)
glm_predicted_num <- predict(glm_fit, test_set)

# use logistic regression prediction to compute accuracy on the test set
glm_accuracy <- accuracy(glm_predicted_num, test_set$disease)

# use glm_accuracy to create accuracy_results table (= Logistic Regression)
accuracy_results <- data_frame(method = "Logistic Regression", accuracy = glm_accuracy)

#############################################################################################################
# Using k-nearest-neighbor (knn) in a first attempt to improve the machine learning algorithm
#############################################################################################################

# compute fit of knn algorithm
set.seed(20, sample.kind="Rounding")
knn_fit <- train(disease ~ ., data = train_set, method = "knn", tuneGrid = data.frame(k = seq(1, 151, 2)))
ggplot(knn_fit, highlight = TRUE)

# calculate predicted presence of heart disease (with knn)
knn_predicted_num <- predict(knn_fit, test_set)

# use knn prediction to compute accuracy on the test set
knn_accuracy <- accuracy(knn_predicted_num, test_set$disease)

# add knn_accuracy to accuracy_results table (= K-Nearest-Neighbor)
accuracy_results <- bind_rows(accuracy_results, data_frame(method = "K-Nearest-Neighbor", 
                                                           accuracy = knn_accuracy))

#############################################################################################################
# Applying a regression tree algorithm to the data which partitions the predictor space 
#############################################################################################################

# compute fit of regression tree algorithm
set.seed(20, sample.kind="Rounding")
rpart_fit <- train(disease ~ ., data = train_set, method = "rpart", 
                   tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)))
ggplot(rpart_fit, highlight = TRUE)

# plot regression tree
plot(rpart_fit$finalModel, margin = 0.1)
text(rpart_fit$finalModel, cex = 0.75)

# calculate predicted presence of heart disease (with rpart)
rpart_predicted_num <- predict(rpart_fit, test_set)

# use regression tree prediction to compute accuracy on the test set
rpart_accuracy <- accuracy(rpart_predicted_num, test_set$disease)

# add rpart_accuracy to accuracy_results table (= Regression Tree)
accuracy_results <- bind_rows(accuracy_results, data_frame(method = "Regression Tree", 
                                                           accuracy = rpart_accuracy))

#############################################################################################################
# Trying to apply a random forest algorithm to overcome the shortcomings of regression trees
#############################################################################################################

# compute fit of random forest algorithm
set.seed(20, sample.kind="Rounding")
rf_fit <- train(disease ~ ., data = train_set, method = "rf", tuneGrid = data.frame(mtry = seq(50, 200, 10)))
ggplot(rf_fit, highlight = TRUE)

# calculate predicted presence of heart disease (with rf)
rf_predicted_num <- predict(rf_fit, test_set)

# use random forest prediction to compute accuracy on the test set
rf_accuracy <- accuracy(rf_predicted_num, test_set$disease)

# add rf_accuracy to accuracy_results table (= Random Forest)
accuracy_results <- bind_rows(accuracy_results, data_frame(method = "Random Forest", 
                                                           accuracy = rf_accuracy))
accuracy_results %>% knitr::kable()
