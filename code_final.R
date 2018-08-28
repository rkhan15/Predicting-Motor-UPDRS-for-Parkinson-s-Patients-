library(RCurl)
library(reshape2)
library(ggplot2)
library(corrplot)
library(leaps)
library(glmnet)
library(MASS)
library(tree)
library(randomForest)

# Data exploration ######################################################################

# Load and clean data (change column names) 
data.park <- read.csv(text = getURL("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"))
head(data.park)
names(data.park)[names(data.park) == 'Jitter...'] <- 'Jitter(%)'
names(data.park)[names(data.park) == 'Jitter.Abs.'] <- 'Jitter(Abs)'
names(data.park)[names(data.park) == 'Shimmer.dB.'] <- 'Shimmer(dB)'
data.park.cols.1 <- data.park[,2:14]
data.park.cols.2 <- data.park[,15:21]
ggplot(melt(data.park.cols.1),aes(x = value)) + 
  facet_wrap(~variable,scales = "free_x") + 
  geom_histogram()
ggplot(melt(data.park.cols.2),aes(x = value)) + 
  facet_wrap(~variable,scales = "free_x") + 
  geom_histogram()

# Plot correlations between predictors 
corrplot(cor(data.park[,2:ncol(data.park)]), type="upper")

# Convert 'sex' to factor
data.park$sex <- factor(data.park$sex, levels = c("0", "1"), labels = c("Male", "Female"))

# Drop 'subject' column
data.park <- data.park[-1]

# Standardize data
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
data.park.norm <- as.data.frame(lapply(data.park[c(1, 3, 5:21)], normalize))
data.park.norm$sex <- data.park$sex
data.park.norm$motor_UPDRS <- data.park$motor_UPDRS

# Divide into training set and test set
seed.num <- which(letters == "r") + which(letters == "k")
set.seed(seed.num)
train <- sample(1:dim(data.park)[1], (dim(data.park)[1]/4)*3)
test <- -train
data.park.train <- data.park.norm[train,]
data.park.test <- data.park.norm[test,]

# Perform multiple linear regression ######################################################################
fit.linear.1 <- lm(motor_UPDRS~. - total_UPDRS, data=data.park.train)
summary(fit.linear.1)

# Best subsets selection for multiple linear regression
best.subsets <- regsubsets(motor_UPDRS~. - total_UPDRS, data=data.park.train)
subsets.summary <- summary(best.subsets)

par(mfrow=c(1,3))
plot(subsets.summary$cp, xlab = "# of Predictors", ylab = "Cp", type = "l")
cp.best <- min(subsets.summary$cp)
points(c(1:10)[subsets.summary$cp == cp.best], cp.best, pch=2, col="red")
abline(v = c(1:10)[subsets.summary$cp == cp.best], lwd=1, lty=2, col="red")

plot(subsets.summary$bic, xlab = "# of Predictors", ylab = "BIC", type = "l")
bic.best <- min(subsets.summary$bic)
points(c(1:10)[subsets.summary$bic == bic.best], bic.best, pch=2, col="red")
abline(v = c(1:10)[subsets.summary$bic == bic.best], lwd=1, lty=2, col="red")

plot(subsets.summary$adjr2, xlab = "# of Predictors", ylab = "Adjusted R Squared", type = "l")
adjr2.best <- max(subsets.summary$adjr2)
points(c(1:10)[subsets.summary$adjr2 == adjr2.best], adjr2.best, pch=2, col="red")
abline(v = c(1:10)[subsets.summary$adjr2 == adjr2.best], lwd=1, lty=2, col="red")
title("Best Subset Selection", outer = TRUE, line=-1)

# Fitting best model
fit.linear.best <- lm(motor_UPDRS ~ age + test_time + Jitter.Abs. + Shimmer.APQ5 + Shimmer.APQ11 + HNR + DFA + PPE, data=data.park.train)
residual.std.error <- summary(fit.linear.best)[[6]]
par(mfrow=c(1,2))
plot(fit.linear.best, which=c(2,1))

linear.pred <- predict(fit.linear.best, data.park.test, type="response")
linear.SSE <- sum((data.park.test$motor_UPDRS - linear.pred) ^ 2)
linear.SST <- sum((data.park.test$motor_UPDRS - mean(data.park.test$motor_UPDRS)) ^ 2)
linear.r.squared <- 1 - linear.SSE/linear.SST
linear.mse <- mean(fit.linear.best$residuals^2)

# Perform ridge regression ######################################################################
train.matrix <- model.matrix(motor_UPDRS ~ . - total_UPDRS, data = data.park.train)
test.matrix <- model.matrix(motor_UPDRS ~ . - total_UPDRS, data = data.park.test)
grid <- 10 ^ seq(4,-2,length=100)

ridge.model <- cv.glmnet(train.matrix, data.park.train$motor_UPDRS, alpha = 0, lambda=grid, thresh=1e-12)
lambda.best <- ridge.model$lambda.min

ridge.pred <- predict(ridge.model, newx = test.matrix, s = lambda.best)
ridge.mse <- mean((data.park.test$motor_UPDRS - ridge.pred)^2)

ridge.SSE <- sum((data.park.test$motor_UPDRS - ridge.pred) ^ 2)
ridge.SST <- sum((data.park.test$motor_UPDRS - mean(data.park.test$motor_UPDRS)) ^ 2)
ridge.r.squared <- 1 - ridge.SSE/ridge.SST

# Perform LASSO regression ######################################################################
data.park.train$sex <- as.numeric(data.park.train$sex)
data.park.test$sex <- as.numeric(data.park.test$sex)
X.all <- as.matrix(data.park.train[, c(1:2, 4:20)])
Y <- as.matrix(data.park.train[21])

# Use CV to select the optimal value of lambda
lasso.cv <- cv.glmnet(X.all, Y, alpha=1)
plot(lasso.cv)
lasso.cv$lambda.min
lasso.model <- glmnet(X.all, Y, alpha=1, lambda=lasso.cv$lambda.min)
coef(lasso.model)[,1]

# Perform Decision Tree regression ######################################################################
tree.motor <- tree(motor_UPDRS ~ . - total_UPDRS, data.park.train)
plot(tree.motor)
text(tree.motor, cex=0.75)
title("Decision Tree to Predict 'motor_UPDRS'")
decision.pred <- predict(tree.motor, data.park.test, type="vector")
decision.SSE <- sum((data.park.test$motor_UPDRS - decision.pred) ^ 2)
decision.SST <- sum((data.park.test$motor_UPDRS - mean(data.park.test$motor_UPDRS)) ^ 2)
decision.r.squared <- 1 - decision.SSE/decision.SST
decision.mse <- mean((data.park.test$motor_UPDRS - decision.pred)^2)

# Perform Random Forest regression ######################################################################
forest.motor <- randomForest(motor_UPDRS ~ . - total_UPDRS, data.park.train)
forest.motor
plot(forest.motor)

# mtry is the # of Variables randomly chosen at each split
# We want to make sure the random forest tries to incorporate all 19 variables
oob.err=double(19)
test.err=double(19)

for(mtry in 1:19) 
{
  forest.motor <- randomForest(motor_UPDRS ~ . - total_UPDRS, data = data.park.train, mtry=mtry,ntree=400)
  oob.err[mtry] <- forest.motor$mse[400] # Error of all Trees fitted
  
  pred <- predict(forest.motor, data.park.test) # Predictions on Test Set for each Tree
  test.err[mtry] <- with(data.park.test, mean( (data.park.test$motor_UPDRS - pred)^2)) # Mean Squared Test Error
  
  cat(mtry," ") # Printing the output to the console
}

test.err # Vector of test error computed above, after adding each variable to model
oob.err # Out-of-bag Estimation

forest.pred <- pred
forest.SSE <- sum((data.park.test$motor_UPDRS - forest.pred) ^ 2)
forest.SST <- sum((data.park.test$motor_UPDRS - mean(data.park.test$motor_UPDRS)) ^ 2)
forest.r.squared <- 1 - forest.SSE/forest.SST

# So here we are growing 400 trees for 19 times (for all 19 predictors).
matplot(1:mtry , cbind(oob.err,test.err), pch=19 , col=c("red","blue"),type="b",ylab="Mean Squared Error",xlab="Number of Predictors Considered at each Split")
legend("topright",legend=c("Out of Bag Error","Test Error"),pch=19, col=c("red","blue"))
# Based on the error plot, the error tends to be minimized at adding all 19 variables.
