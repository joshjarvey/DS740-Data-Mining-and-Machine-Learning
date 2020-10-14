# define data frame heart as in WebWork Lesson 8, Problem 4

input = read.csv("Heart_Disease_Cleveland.csv")
names(input)
heart = input[,c(1,4,5,8,10)]
heart$HD = rep(0, length(input$DiseaseStatus))
heart$HD[which(input$DiseaseStatus > 0)] = 1
heart$HD = factor(heart$HD)

# read in libraries
library(MASS)

##### model assessment OUTER 10-fold CV (with model selection INNER 10-fold CV as part of model-fitting) #####

xy.out = heart
n.out = dim(xy.out)[1]

#define the cross-validation splits 
k.out = 10 
groups.out = c(rep(1:k.out,floor(n.out/k.out)),1:(n.out%%k.out))  #produces list of group labels
set.seed(8, sample.kind = "Rounding")
cvgroups.out = sample(groups.out,n.out)  #orders randomly, with seed (8) 

allpredictedCV.out = rep(NA,n.out)

##### model assessment OUTER shell #####
for (j in 1:k.out)  {  #be careful not to re-use loop indices
  groupj.out = (cvgroups.out == j)

  # define the training set for outer loop
  trainxy.out = xy.out[!groupj.out,]
  
  #define the validation set for outer loop
  testxy.out = xy.out[groupj.out,]

  ##############################################
  ###   model selection on trainxy.out       ###
  ##############################################
  ##entire model-fitting process##
  xy.in = trainxy.out  # fixed to be fit ONLY to the training data from the outer split
  n.in = dim(xy.in)[1]
  ncv = 10
  if ((n.in%%ncv) == 0) {
    groups.in= rep(1:ncv,floor(n.in/ncv))} else {
      groups.in=c(rep(1:ncv,floor(n.in/ncv)),(1:(n.in%%ncv)))
    }
  
  cvgroups.in = sample(groups.in,n.in)
  # set up storage
  allpredictedcv10 = matrix(,ncol=6,nrow=n.in)
  
  # with model selection 
  for (i in 1:ncv) {
    newdata.in = xy.in[cvgroups.in==i,]
    
    lda2fit = lda(HD ~ MaxHeartRate + STdepress, data=xy.in, subset=(cvgroups.in!=i))
    allpredictedcv10[cvgroups.in==i,1] = predict(lda2fit,newdata.in)$class

    lda5fit = lda(HD ~., data= xy.in, subset=(cvgroups.in!=i))
    allpredictedcv10[cvgroups.in==i,2] = predict(lda5fit,newdata.in)$class
    
    qda2fit = qda(HD ~ MaxHeartRate + STdepress, data=xy.in, subset=(cvgroups.in!=i))
    allpredictedcv10[cvgroups.in==i,3] = predict(qda2fit,newdata.in)$class
    
    qda5fit = qda(HD ~., data= xy.in, subset=(cvgroups.in!=i))
    allpredictedcv10[cvgroups.in==i,4] = predict(qda5fit,newdata.in)$class
    
    log2fit = glm(HD ~ MaxHeartRate + STdepress, data=xy.in, subset=(cvgroups.in!=i), family=binomial)
    log2prob = predict(log2fit,newdata.in,type="response")
    log2fact = rep(1,dim(newdata.in)[1]); log2fact[log2prob > 0.5] = 2
    allpredictedcv10[cvgroups.in==i,5] = log2fact
    
    log5fit = glm(HD ~., data= xy.in, subset=(cvgroups.in!=i),family=binomial)
    log5prob = predict(log5fit,newdata.in,type="response")
    log5fact = rep(1,dim(newdata.in)[1]); log5fact[log5prob > 0.5] = 2
    allpredictedcv10[cvgroups.in==i,6] = log5fact
  }
  #relabel as original values, not factor levels
  allpredictedcv10 = allpredictedcv10-1  # now a table of predicted 0-1 values for HD
  
  #compute the CV values
  allcv10 = rep(0,6)
  for (m in 1:6) allcv10[m] = sum(xy.in$HD!=allpredictedcv10[,m])/n.in
  bestmodels = (1:6)[allcv10 == min(allcv10)]
  ##############################################
  ###   resulting in bestmodels              ###
  ##############################################

  bestmodel = ifelse(length(bestmodels)==1,bestmodels,sample(bestmodels,1))
  print(allcv10)
  print(paste("Best model at outer loop",j,"is",bestmodel))
  #some code-checking assistance:
  #print(j)
  #print(allcv10.in)
  #print(bestmodels)
  #print(bestmodel)

  if (bestmodel == 1)  {
    lda2fit.train = lda(HD ~ MaxHeartRate + STdepress, data=trainxy.out)
    predictvalid = as.numeric(predict(lda2fit.train, testxy.out)$class)
  }
  if (bestmodel == 2)  {
    lda5fit.train = lda(HD ~ ., data=trainxy.out)
    predictvalid = as.numeric(predict(lda5fit.train, testxy.out)$class)
  }
  if (bestmodel == 3)  {
    qda2fit.train = qda(HD ~ MaxHeartRate + STdepress, data=trainxy.out)
    predictvalid = as.numeric(predict(qda2fit.train, testxy.out)$class)
  }
  if (bestmodel == 4)  {
    qda5fit.train = qda(HD ~ ., data=trainxy.out)
    predictvalid = as.numeric(predict(qda5fit.train, testxy.out)$class)
  }
  if (bestmodel == 5)  {
    log2fit.train = glm(HD ~ MaxHeartRate + STdepress, data= trainxy.out, family=binomial)
    log2prob.test = predict(log2fit.train,testxy.out,type="response")
    predictvalid = rep(1,dim(testxy.out)[1]); predictvalid[log2prob.test > 0.5] = 2
  }
  if (bestmodel == 6)  {
    log5fit.train = glm(HD ~ ., data= trainxy.out, family=binomial)
    log5prob.test = predict(log5fit.train,testxy.out,type="response")
    predictvalid = rep(1,dim(testxy.out)[1]); predictvalid[log5prob.test > 0.5] = 2
  }
  #relabel as original values, not factor levels
  predictvalid = predictvalid-1  # now a vector of predicted 0-1 values for HD in validation set
  
  allpredictedCV.out[groupj.out] = predictvalid

}

# the output shows the different models selected in the outer loop - purpose is only to observe processing
# however, the model selection was done previously (in Problem 4) via single-level cross-validation

#Purpose of double cross-validation:
# assessment - what proportion of the cross-validated classifications (valid predictions of 
# new observations, based on model selected using the entire model-selection process)
# match the actual observations?
table(heart$HD,allpredictedCV.out)
CV10.out = sum(heart$HD!=allpredictedCV.out)/n.out
p.out = 1-CV10.out; p.out  # (133+89)/303

# this sounds pretty reasonable; but note that just always GUESSING the majority 
# classification, 0, would result in a proportion correctly classified of 0.541... 
table(heart$HD)/n.out
# so (cross-validated) proportion 0.733 of correct classifications  is an improvement, 
# but not a dramatic one
