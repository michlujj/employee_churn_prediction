## Employee's Churn Prediction

# to clear R environment
rm(list= ls())

library(tidyverse) # data manipulation and ggplot
library(magrittr) # to run the %>% operator

#install.packages("devtools")
#devtools::install_github("r-lib/conflicted")

library(conflicted)
library(dplyr)

# to import dataset
df <- read.csv('C:/Users/miche/Desktop/machine_learning/WA_Fn-UseC_-HR-Employee-Attrition.csv')

head(df)

# to check if variables are in their correct data types
glimpse(df)

# to check for missing variables
sapply(df, function(x) sum(is.na(x)))

# there are no duplicated values in dataset
duplicated(df)
sum(duplicated(df))

# to see distribution of target variable 'Attrition'
ggplot(df, aes(x=Attrition)) + 
  geom_bar(fill='orange')+
  ggtitle('Attrition count of employees')

# to see actual numbers of employees who left the company
attrition_df <- df %>% group_by(Attrition) %>% summarise(count=n()) %>% as.data.frame()
attrition_df

summary(df$Age)

# to maximax age is 60 years
boxplot(df$Age, fill='pink')

# to use custom function to separate 'Age' into different categories
df <- df %>% mutate(age_categories = case_when (Age >= 18 & Age <= 30 ~ "18 to 30",
                                          Age >= 31 & Age <= 45 ~ "31 to 45",
                                          Age >= 46 & Age <= 65~ "46 to 60"))

# to see new column
head(df)

# to see count of gender by age groups
ggplot(df, aes(x=age_categories, fill=Gender))+
  geom_bar()+
  ggtitle('Gender by Age groups')

df %>% group_by(Gender,age_categories) %>% summarise(count=n())

# to see Attrition count by age categories
ggplot(df, aes(x=age_categories, fill=Attrition))+
  geom_bar()+
  ggtitle('Attrition count by Age groups')

df %>% group_by(age_categories, Attrition) %>%summarise(count=n())

# to see Attrition count by Job title
ggplot(df, aes(y=JobRole, fill=Attrition))+
  geom_bar()+
  ggtitle('Attrition count by Job titles')

df %>% group_by(JobRole, Attrition) %>%summarise(count=n()) %>% as.data.frame()

# to see Attrition count by Gender
ggplot(df, aes(x=Gender, fill=Attrition))+
  geom_bar()+
  ggtitle('Attrition count by Job titles')

df %>% group_by(Gender, Attrition) %>%summarise(count=n())

# to see Attrition count by Marital status
ggplot(df, aes(x=MaritalStatus, fill=Attrition))+
  geom_bar()+
  ggtitle('Attrition count by Marital Status')

# it is observed that people who left the company are mostly married
df %>% group_by(MaritalStatus, Attrition) %>%summarise(count=n())


# to see Attrition count by Job Satisfaction
ggplot(df, aes(x=JobSatisfaction, fill=Attrition))+
  geom_bar()+
  ggtitle('Attrition count by Job Satisfaction')

# to see Attrition count by Environment Satisfaction
ggplot(df, aes(x=EnvironmentSatisfaction, fill=Attrition))+
  geom_bar()+
  ggtitle('Attrition count by Job Satisfaction')

# to see Attrition count by Environment Satisfaction
ggplot(df, aes(x=RelationshipSatisfaction, fill=Attrition))+
  geom_bar()+
  ggtitle('Relationship Satisfaction by Job Satisfaction')

# to see Attrition count by Education field
ggplot(df, aes(y=EducationField, fill=Attrition))+
  geom_bar()+
  ggtitle('Education field by Job Satisfaction')

# to see Marital Status by Age categories
ggplot(df, aes(x=MaritalStatus, fill=age_categories))+
  geom_bar()+
  ggtitle('Marital Status by Age groups')

# to see Attrition count by Overtime
ggplot(df, aes(x=OverTime, fill=Attrition))+
  geom_bar()+
  ggtitle('Attrition count by Working Overtime')

# to see Attrition count by Business Travel
ggplot(df, aes(x=BusinessTravel, fill=Attrition))+
  geom_bar()+
  ggtitle('Attrition count by Business Travel')

glimpse(df)

# to remove unimportant variables before predictive modelling
df <- select(df, -c(age_categories, EmployeeCount, EmployeeNumber,Over18,StandardHours))

# to convert 'Attrition' as a factor as it is a response variable
df$Attrition <- as.factor(df$Attrition)

# to convert all categorical variables into factors
# 'BusinessTravel',' EducationField','Gender','JobRole'
#  'MaritalStatus','OverTime'
df$BusinessTravel <- as.factor(df$BusinessTravel)
df$Department <- as.factor(df$Department)
df$EducationField <- as.factor(df$EducationField)
df$Gender <- as.factor(df$Gender)
df$JobRole <- as.factor(df$JobRole)
df$MaritalStatus <- as.factor(df$MaritalStatus)
df$OverTime <- as.factor(df$OverTime)

#to create Training and Test data 
# using createDataPartition from 'Caret' Package
library(caret)
set.seed(101)

# training data = 70%
trainDataIndex <- createDataPartition(df$Attrition, p=0.7, list = FALSE)  # 70% training data
trainData <- df[trainDataIndex, ]
testData <- df[-trainDataIndex, ]

library(devtools)
#install_github("cran/DMwR")

# to load DMwR library installed from github
library(DMwR)

# to oversample "Attrition" target variable using SMOTE
# to set perc.over=100 to double quantity of positive cases
# set perc.under=200 to keep half of what was created as neg cases
trainData$Attrition <- as.factor(trainData$Attrition)
trainData <- SMOTE(Attrition ~ ., trainData, perc.over = 100, perc.under=200)

# to check if the 'Reveue' TRUE and FALSE  are in equal ratios
table(trainData$Attrition)


# to train the Logistic regression model
log.model <- glm(formula=Attrition ~ . , family = binomial(link='logit'),data = df)

# print the results of logistic regression model
summary(log.model)


# to import the decision tree classification algorithm
library(rpart)

# to train the decision tree
tree <- rpart(Attrition ~.,method='class',data = trainData)

# to predict the Attrition label on the test data
tree_preds <- predict(tree, testData)

# to check the predicted values for Decision tree
head(tree_preds)

# Turn these two columns into one column to match the original Yes/No Label for Attrition column
tree_preds <- as.data.frame(tree_preds)

joiner <- function(x){
  if (x>=0.5){
    return('Yes')
  }else{
    return("No")
  }
}

tree_preds$Attrition <- sapply(tree_preds$Yes,joiner)

head(tree_preds)

# to print confusion matrix for decision tree
table(tree_preds$Attrition, testData$Attrition)

# to plot out the decision tree
library(rpart.plot)

prp(tree)

# to load machine learning algorithm, a Classification prediction
library(randomForest)

# a set of tools to help explain which variables are most important in RF
library(randomForestExplainer)

x <-trainData

# Create model with default paramters, estimated accuracy is 91.18%
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(Attrition~., data=trainData, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)

# Random Search for mtry using CARET package
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(seed)
mtry <- sqrt(ncol(x))
rf_random <- train(Attrition~., data=trainData, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

# to build the random forest model again using best mtry value = 12
set.seed(101)
rf <-randomForest(Attrition~.,data=trainData, mtry=12, importance=TRUE,ntree=500)
rf

# to display the cross-validation error rate against the number of trees
plot(rf)

# to examine the variable importance of predictors in Random Forest model
importance(rf)

# print in order of Variable Importance
varImpPlot(rf, sort=TRUE, main = 'Features Importance by RF')

# prediction and confusion matrix for testing data
p2 <- predict (rf,testData)

# to generated a confusion matrix on the testing data
# Random Forest classifer accuracy rate = 80%
confusionMatrix(p2, testData$Attrition)

















