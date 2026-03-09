# =========================================================
# BANK MARKETING MACHINE LEARNING ANALYSIS
# MBA / Data Science Assignment
# =========================================================


# ---------------------------------------------------------
# 1. Install Required Packages (Run once if not installed)
# ---------------------------------------------------------

packages <- c("tidyverse","caret","e1071","rpart","rpart.plot",
              "class","pROC","corrplot")

install.packages(setdiff(packages, rownames(installed.packages())))


# ---------------------------------------------------------
# 2. Load Libraries
# ---------------------------------------------------------

library(tidyverse)
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(class)
library(pROC)
library(corrplot)


# ---------------------------------------------------------
# 3. Load Dataset
# ---------------------------------------------------------

bank <- read.csv(file.choose(), sep=";")

str(bank)
summary(bank)


# ---------------------------------------------------------
# 4. Convert Character Variables to Factors
# ---------------------------------------------------------

bank <- bank %>%
  mutate(across(where(is.character), as.factor))

str(bank)


# ---------------------------------------------------------
# 5. Target Variable Distribution
# ---------------------------------------------------------

table(bank$y)
prop.table(table(bank$y))


# ---------------------------------------------------------
# 6. Descriptive Statistics (Numeric Variables)
# ---------------------------------------------------------

numeric_vars <- c("age","balance","duration",
                  "campaign","pdays","previous")

summary(bank[numeric_vars])


# ---------------------------------------------------------
# 7. Exploratory Data Analysis (Distribution Plots)
# ---------------------------------------------------------

bank %>%
  select(all_of(numeric_vars)) %>%
  pivot_longer(cols = everything()) %>%
  ggplot(aes(x=value)) +
  geom_histogram(bins=30, fill="steelblue", color="white") +
  facet_wrap(~name, scales="free") +
  theme_minimal() +
  labs(title="Distribution of Numeric Variables")


# ---------------------------------------------------------
# 8. Correlation Analysis
# ---------------------------------------------------------

cor_pearson <- cor(bank[numeric_vars], method="pearson")
round(cor_pearson,3)

cor_spearman <- cor(bank[numeric_vars], method="spearman")
round(cor_spearman,3)


# ---------------------------------------------------------
# 9. Correlation Heatmap
# ---------------------------------------------------------

corrplot(cor_pearson, method="color", type="upper")


# ---------------------------------------------------------
# 10. Train-Test Split (70% Training / 30% Testing)
# ---------------------------------------------------------

set.seed(123)

train_index <- createDataPartition(bank$y, p=0.7, list=FALSE)

train <- bank[train_index,]
test <- bank[-train_index,]

prop.table(table(train$y))
prop.table(table(test$y))


# ---------------------------------------------------------
# 11. Hypothesis Testing (Call Duration Difference)
# ---------------------------------------------------------

yes_duration <- train$duration[train$y=="yes"]
no_duration <- train$duration[train$y=="no"]

t_test_result <- t.test(yes_duration, no_duration)

t_test_result


# ---------------------------------------------------------
# 12. Chi-Square Test
# ---------------------------------------------------------

tbl_poutcome <- table(train$poutcome, train$y)

chisq.test(tbl_poutcome)


# ---------------------------------------------------------
# 13. Create Dummy Variables (Feature Engineering)
# ---------------------------------------------------------

dummies <- dummyVars(y ~ ., data=train)

train_x <- predict(dummies,newdata=train) %>% as.data.frame()
test_x <- predict(dummies,newdata=test) %>% as.data.frame()

train_y <- train$y
test_y <- test$y


# ---------------------------------------------------------
# 14. Feature Scaling (For KNN)
# ---------------------------------------------------------

preproc <- preProcess(train_x, method=c("center","scale"))

train_x_scaled <- predict(preproc, train_x)
test_x_scaled <- predict(preproc, test_x)


# ---------------------------------------------------------
# 15. Logistic Regression Model
# ---------------------------------------------------------

logit_model <- glm(y ~ ., data=train, family="binomial")

summary(logit_model)


# ---------------------------------------------------------
# 16. Logistic Regression Predictions
# ---------------------------------------------------------

logit_prob <- predict(logit_model, newdata=test, type="response")

logit_pred <- ifelse(logit_prob > 0.5, "yes","no") %>%
  factor(levels=c("no","yes"))


# ---------------------------------------------------------
# 17. Decision Tree Model
# ---------------------------------------------------------

tree_model <- rpart(y ~ ., data=train, method="class")

rpart.plot(tree_model)


# ---------------------------------------------------------
# 18. Naive Bayes Model
# ---------------------------------------------------------

nb_model <- naiveBayes(y ~ ., data=train)

nb_pred <- predict(nb_model, newdata=test)


# ---------------------------------------------------------
# 19. K-Nearest Neighbors (KNN)
# ---------------------------------------------------------

k <- 5

knn_pred <- knn(
  train=train_x_scaled,
  test=test_x_scaled,
  cl=train_y,
  k=k
)


# ---------------------------------------------------------
# 20. Model Evaluation
# ---------------------------------------------------------

# Logistic Regression
confusionMatrix(logit_pred, test_y, positive="yes")

# Decision Tree
tree_pred <- predict(tree_model, newdata=test, type="class")
confusionMatrix(tree_pred, test_y, positive="yes")

# Naive Bayes
confusionMatrix(nb_pred, test_y, positive="yes")

# KNN
confusionMatrix(knn_pred, test_y, positive="yes")


# ---------------------------------------------------------
# 21. ROC Curve and AUC
# ---------------------------------------------------------

roc_logit <- roc(test_y, logit_prob)

plot(roc_logit, col="blue", main="ROC Curve - Logistic Regression")

auc(roc_logit)


# ---------------------------------------------------------
# 22. Prepare Data for Cross Validation
# ---------------------------------------------------------

train_y <- factor(train_y, levels=c("no","yes"))


# ---------------------------------------------------------
# 23. K-Fold Cross Validation (5-Fold)
# ---------------------------------------------------------

set.seed(123)

train_control <- trainControl(
  method="cv",
  number=5,
  classProbs=TRUE,
  summaryFunction=twoClassSummary
)

logit_cv <- train(
  x=train_x,
  y=train_y,
  method="glm",
  family="binomial",
  trControl=train_control,
  metric="ROC"
)

print(logit_cv)