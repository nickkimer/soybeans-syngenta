setwd("~/Desktop/Fall 2016 Semester/SYS 6018/Final Project")
library(dplyr)
library(mlbench) # H2o Deep Learning
library(h2o)  # H2o Deep Learning: Also need to install RCurl and rjson

# save(training_data, file= "training_data.Rda")
# save(filtered_data, file= "filtered_data.Rda")

load("filtered_data.Rda")
load("training_data.Rda")
# training_data = read.csv('EXPERIMENT DATA.csv',header=TRUE)
# eval_data= read.csv('EVALUATION SET.csv',header=TRUE) 
# eval_data$VARIETY %in% Varitey_names_yes
# Varitey_names_yes= as.character(unique(training_data[training_data$GRAD=="YES",]$VARIETY))
# training_data$NEW_GRAD = NA
# for (i in 1:nrow(training_data)){ 
#   if (training_data$VARIETY[i] %in% Varitey_names_yes){ 
#     training_data$NEW_GRAD[i]= "YES"}
#   else {training_data$NEW_GRAD[i]= 'NO'}}

# training_data$NEW_GRAD= as.factor(training_data$NEW_GRAD)
# 
# filtered_data= training_data[-which(training_data$CLASS_OF %in% c("2013")),]

test.data = training_data[which(training_data$CLASS_OF %in% c("2013")),] 
test.data = test.data[-which(test.data$CLASS_OF %in% "."),]

to_predict= test.data[which(test.data$CLASS_OF %in% "2014"),]


# training_data$LOCATION = factor(training_data$LOCATION)
# training_data$FAMILY = factor(training_data$FAMILY)
# training_data$VARIETY = factor(training_data$VARIETY)
# training_data$GRAD = factor(training_data$GRAD)

########################################################################################################################
###################################                Deep Learning             ###########################################
########################################################################################################################
# initialize h2o server
h2o = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)
training.data.h2o = as.h2o(filtered_data, destination_frame = "training_data")


h2o.model_complete = h2o.deeplearning(x = 2:9,  
                                         y = 13,
                                         training.data.h2o,
                                         activation = "TanhWithDropout", 
                                         input_dropout_ratio = 0.2,
                                         hidden_dropout_ratios = c(0.5,0.5,0.5),
                                         balance_classes = FALSE, 
                                         hidden = c(50,50,50),
                                         #nfolds= 5,
                                         epochs = 100)
######################## Test Deep Learning Model with all Predictors  ##########################

test.data.h2o = as.h2o(test.data, destination_frame = 'testing.data')
h2o.complete_predictions = h2o.predict(h2o.model_complete, test.data.h2o)
h2o.complete_predictions = as.data.frame(h2o.complete_predictions)$predict
predict =  h2o.predict(h2o.model_complete, test.data.h2o)
perf_deep_complete <- h2o.performance(h2o.model_complete, newdata = test.data.h2o)

summary(h2o.model_complete)
perf_deep_complete
table(h2o.complete_predictions)
########ROC
plot(perf_deep_complete,type = 'roc', col="blue", typ ='b')

# show accuracy
sum(test.data[, 1] != h2o.complete_predictions) / length(h2o.complete_predictions)

########################################################################################################################
###############################                Deep Learning Model Tuning             ##################################
########################################################################################################################

# ##Tuning model
models <- c()

for (i in 1:10) {
  rand_activation <- c("TanhWithDropout", "RectifierWithDropout")[sample(1:2,1)]
  rand_numlayers <- sample(2:5,1)
  rand_hidden <- c(sample(10:50,rand_numlayers,T))
  rand_l1 <- runif(1, 0, 1e-3)
  rand_l2 <- runif(1, 0, 1e-3)
  rand_dropout <- c(runif(rand_numlayers, 0, 0.6))
  rand_input_dropout <- runif(1, 0, 0.5)
  dlmodel <- h2o.deeplearning(x=2:17, y=1, training.data.h2o, validation_frame = test.data.h2o, epochs=0.1,
                              activation=rand_activation, hidden=rand_hidden, l1=rand_l1, l2=rand_l2,
                              input_dropout_ratio=rand_input_dropout, hidden_dropout_ratios=rand_dropout)
  models <- c(models, dlmodel)
}
dlmodel
best_err <- models[[1]]@model$validation_metrics@metrics$MSE #best model from grid search above
for (i in 1:length(models)) {
  err <- models[[i]]@model$validation_metrics@metrics$MSE
  if (err < best_err) {
    best_err <- err
    best_model <- models[[i]]
  }
}

best_model@model_id
best_params <- best_model@allparameters
best_params$activation
best_params$hidden
best_params$l1
best_params$l2
best_params$input_dropout_ratio
best_params$hidden_dropout_ratios

dlmodel_continued <- h2o.deeplearning(x = 2:17, y = 1 , activation = "RectifierWithDropout" , hidden = c(24, 45, 16, 47, 23), training.data.h2o, validation_frame = test.data.h2o, checkpoint = best_model@model_id, l1=best_params$l1, l2=best_params$l2, epochs=0.25)
dlmodel_continued <- h2o.deeplearning(x = 2:17, y = 1 , training.data.h2o, activation = best_params$activation, 
                                      validation_frame = test.data.h2o, l1=best_params$l1, l2=best_params$l2, epochs=0.25)

dlmodel_continued@model$validation_metrics@metrics$AUC


######################## Test Deep Learning Model   ##########################


test.data.h2o = as.h2o(test.data, destination_frame = 'testing.data')
h2o.complete_predictions = h2o.predict(dlmodel_continued, test.data.h2o)
h2o.complete_predictions = as.data.frame(h2o.complete_predictions)$predict
predict =  h2o.predict(dlmodel_continued, test.data.h2o)
perf_deep_complete <- h2o.performance(dlmodel_continued, newdata = test.data.h2o)

summary(h2o.model_complete)
perf_deep_complete
table(h2o.complete_predictions)
########ROC
plot(perf_deep_complete,type = 'roc', col="blue", typ ='b')

# show accuracy
sum(test.data[, 1] != lmodel_continued) / length(lmodel_continued)



















#### DATA CLEANING 
co2011 = data[data$CLASS_OF == "2011", ]
years = c(2009,2010,2011)
co2011$YEAR %>% unique
co2011 = data[data$YEAR %in% years,]
co2011 = co2011[co2011$GRAD != ".",]
co2011$BAGSOLD = NULL
co2011$YEAR = NULL
co2011$CHECK = NULL
co2011$EXPERIMENT = NULL
co2011$REPNO = NULL
co2011$CLASS_OF = NULL
# co2011$FAMILY = NULL

co2011$LOCATION = factor(co2011$LOCATION)
co2011$FAMILY = factor(co2011$FAMILY)
co2011$VARIETY = factor(co2011$VARIETY)
co2011$GRAD = factor(co2011$GRAD)
# Build Classifier to Determine Which Variety Graduates 
# library(bestglm)
# xy = within(co2011, {
#   YEAR = NULL
#   CHECK = NULL
#   EXPERIMENT = NULL
#   REPNO = NULL
#   CLASS_OF = NULL
#   y = GRAD
#   GRAD = NULL
# })
# 
# out = bestglm(xy,family = binomial, IC="AIC",method = 'exhaustive')
# out$BestModels
# summary(out$BestModel)

#### Trying out SVM in classification
install.packages("e1071")
library(e1071)
samp_size = floor(0.75*nrow(co2011))

set.seed(1000)
train_index = sample(seq_len(nrow(co2011)), size = samp_size)

train = co2011[train_index,] #training set
test = co2011[-train_index,] #test set
test_labels = test$GRAD

table(test$GRAD)
table(train$GRAD)

svm_model <- svm(GRAD ~ ., data=train,probability=TRUE)
summary(svm_model)


library(ROCR)
x.svm.prob = predict(svm_model,test,type='prob',probability=T)
table(x.svm.prob,test$GRAD)
x.svm.prob.rocr = prediction(attr(x.svm.prob,"probabilities")[,2],test$GRAD)
x.svm.perf = performance(x.svm.prob.rocr, measure = "tpr",x.measure = "fpr")


plot(x.svm.perf, colorize=TRUE,main="ROC - AUC: ______")
abline(a=0,b=1)
as.numeric(performance(x.svm.prob.rocr, "auc")@y.values)

################################### Mixed Effects Model

library(lme4)
# install.packages("lmerTest")
library(lmerTest)

m1 <- lmer(YIELD ~ GRAD + RM + (1|FAMILY)+(RM|VARIETY)+(1|LOCATION),data=train )
m2 <- lmer(YIELD ~ GRAD + RM +(RM|VARIETY)+(1|LOCATION),data=train )
#Use variety as a fixed effect not a random effect in the mixed effects model
anova(m1,m2)

#Blocking factors should precede experimental factors
m1 <- lmer(YIELD ~ GRAD + RM + VARIETY + (1|FAMILY) + (1|LOCATION),data=train )
summary(m1)


