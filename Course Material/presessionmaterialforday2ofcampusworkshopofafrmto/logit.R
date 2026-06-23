rm(list=ls())
setwd("G:\\My Drive\\Asus\\Subjects_Preparation\\NuLearn\\AFRM\\Batch_11\\Sessions\\CampusWS_2")

data=read.table("logit.csv", header=T, sep=",")
train_data=data[1:500,]
test_data=data[501:701,]
#Logit Model
#Fit Training Data
mylogit <- glm(y~.,train_data, family = binomial(link = "logit"))
summary(mylogit)

#Predict the default
preds = predict(mylogit , test_data , type = 'response')
preds

preds_def = ifelse(preds >= 0.50 , 1 , 0)
accuracy=mean(preds_def==test_data$y)
accuracy


