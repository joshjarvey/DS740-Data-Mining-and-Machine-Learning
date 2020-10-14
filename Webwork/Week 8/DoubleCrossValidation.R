  #read in the dataset
bodyfat = read.csv("C:/Users/joshj/Documents/DS740-Data Mining and Machine Learning/Webwork/Week 8/bodyfat.csv")

  #count number of samples
n = nrow(bodyfat)

  #creating the linear models to consider
LinModel1 = (BodyFatSiri ~ Abs)
LinModel2 = (BodyFatSiri ~ Abs+Weight)
LinModel3 = (BodyFatSiri ~ Abs+Weight+Wrist+Forearm)
LinModel4 = (BodyFatSiri ~ Abs+Weight+Wrist+Forearm+Neck+Biceps+Age)
LinModel5 = (BodyFatSiri ~ Abs+Weight+Wrist+Forearm+Neck+Biceps+Age+Thigh+Hip+Ankle)
LinModel6 = (BodyFatSiri ~ Abs+Weight+Wrist+Forearm+Neck+Biceps+Age+Thigh+Hip+Ankle+BMI+Height+Chest+Knee)

  #storing the linear models in a list so they can be referenced later
allLinModels = list(LinModel1,LinModel2,LinModel3,LinModel4,LinModel5,LinModel6)
  #counting the length of the list
nLinmodels = length(allLinModels)


library(glmnet) 
  #consider a list of RR models with these lambda values
lambdalistRR = c(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)
  #consider a list of LASSO models with these lambda values
lambdalistLASSO = c(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)

  #count the number of models from each RR and LASSO
nRRmodels = length(lambdalistRR)
nLASSOmodels = length(lambdalistLASSO)

  #sum up the total number of models to we'll be iterating through
nmodels = nLinmodels+nRRmodels+nLASSOmodels


###################################################################
##### Double cross-validation for modeling-process assessment #####				 
###################################################################


#############################################
##### General prep for outer CV process #####
#############################################

  #select the full dataset into a variables
fulldata.out = bodyfat
  #set the number of Outer folds for our 2x CV.
k.out = 10 
  #count the number of samples in the full dataset.
n.out = nrow(fulldata.out)
  #setting up the list of indices for CV groups based on the dataset. 
  #this first line replicates 1:number of folds, an even amount of times. 
groups.out = c(rep(1:k.out,floor(n.out/k.out)))
  #however, that might not always work where the data set is even, and if so, append the "remainder" to the indice group. 
if(floor(n.out/k.out) != (n.out/k.out)){
  groups.out = c(groups.out, 1:(n.out%%k.out))
}

  #set seed for reproducability
set.seed(8, sample.kind = "Rounding")
  #randomize the indices using sample()
cvgroups.out = sample(groups.out,n.out)

  #set up storage for predicted values from the double-cross-validation
allpredictedCV.out = rep(NA,n.out)
  #set up storage to see what models are "best" on the inner loops
allbestmodels = rep(NA,k.out)

####################################
###   Begin double-CV process    ###
####################################
for (j in 1:k.out){ 

  ################################################################################
  ##                    ...Prep for OUTER CV process...                         ##
  ## (splitting the full dataset into a train/test split for current iteration) ##
  ##        (j'th training split will get passed into inner CV process)         ##
  ################################################################################
    #lets find the indices of the current outer fold's iteration. So if j=1, then find where all the 1's are in the randomized indice set
  groupj.out = (cvgroups.out == j)

    #using this vector as described above, select everything else that's NOT in current fold - this is our "training" data.
  traindata.out = bodyfat[!groupj.out,]
    #using this vector as described above, select ONLY the current fold indices - this is our "test" data
  validdata.out = bodyfat[groupj.out,]
  
  
    #from the training data set, we can create the matrix of x's for training
  trainx.out = model.matrix(BodyFatSiri~.,data=traindata.out)[,-(1:4)]
    #from the testing data set, we can create the matrix of x's for testing
  validx.out = model.matrix(BodyFatSiri~.,data=validdata.out)[,-(1:4)]
    #from the training data set, we can create the vector of y's for training
  trainy.out = traindata.out[,3]
    #from the testing data set, we can create the vector of y's for testing
  validy.out = validdata.out[,3]
  
  ####################################
  ### pipe current training data   ###
  ####################################
  ########	:	:	:	:	:	:	:  ###########
  
    #now that we've split up our outer CV10 data, we can pass in the current iterations "training" dataset
    #if this is iteration j=1, then traindata.out is all of the data from folds 9-10. etc.
  fulldata.in = traindata.out
  
  ########	:	:	:	:	:	:	:  ###########
  ####################################
  ####  start inner CV process    ####
  ####################################
  
      ################################################################################
      ##                    ...Prep for INNER CV process...                         ##
      ## (splitting piped in data set (train from outer) into train/test split)     ##
      ################################################################################
  
    #number folds and groups for (inner) cross-validation for model-selection
  k.in = 10   
    #set sample size of the "inner" dataset. 
  n.in = nrow(fulldata.in)
    #set the indices of the inner groups 
  groups.in = c(rep(1:k.in,floor(n.in/k.in))) 
    #if the groups do not divide evenly, then append the remainder of indices onto the vector
  if(floor(n.in/k.in) != (n.in/k.in)){
    groups.in = c(groups.in, 1:(n.in%%k.in))  
  }
    #now we must randomize the indices of this inner training set.
  cvgroups.in = sample(groups.in,n.in)
    #place-holder for results of CV values from each model. There are 30 models, so 30 results
    #note: this will get replaced 10x times for each outer CV loop.
  allmodelCV.in = rep(NA,nmodels)  

      ####################################
      #### inner CV for linear models  ###
      ####################################
  
    #since linear regression does not have any automatic CV output,
    #set up storage for predicted values from the CV splits, across all linear models.
  allpredictedCV.in = matrix(rep(NA,n.in*nLinmodels),ncol=nLinmodels)
  
  #cycle through all folds:  fit the model to training data, predict test data,
  # and store the (cross-validated) predicted values
  for (i in 1:k.in)  {
    train.in = (cvgroups.in != i)
    test.in = (cvgroups.in == i)
    #fit each of the linear regression models on training, and predict the test
    for (m in 1:nLinmodels) {
      lmfitCV.in = lm(formula = allLinModels[[m]],data=bodyfat,subset=train.in)
      allpredictedCV.in[test.in,m] = predict.lm(lmfitCV.in,fulldata.in[test.in,])
    }
  }
  # compute and store the CV(10) values
  for (m in 1:nLinmodels) { 
    allmodelCV.in[m] = mean((allpredictedCV.in[,m]-fulldata.in$BodyFatSiri)^2)
  }

     #######################################
     #### inner CV for penalized models  ###
     #######################################
    #create the x's of this inner dataset
  x.in = model.matrix(BodyFatSiri~.,data=fulldata.in)[,-(1:4)]
    #create the y's of this inner dataset.
  y.in = fulldata.in[,3]
    
    #RR cross-validation - uses internal cross-validation function
  cvRRglm.in = cv.glmnet(x.in, y.in, lambda=lambdalistRR, alpha = 0, nfolds=k.in, foldid=cvgroups.in)
    #LASSO cross-validation - uses internal cross-validation function
  cvLASSOglm.in = cv.glmnet(x.in, y.in, lambda=lambdalistLASSO, alpha = 1, nfolds=k.in, foldid=cvgroups.in)
  
    #store CV(10) values, in same numeric order as lambda, in storage spots for CV values
  allmodelCV.in[(1:nRRmodels)+nLinmodels] = cvRRglm.in$cvm[order(cvRRglm.in$lambda)]
    #store CV(10) values, in same numeric order as lambda, in storage spots for CV values
  allmodelCV.in[(1:nLASSOmodels)+nRRmodels+nLinmodels] = cvLASSOglm.in$cvm[order(cvLASSOglm.in$lambda)]

     ############################################
     ## End model fitting inner CV calculation ##
     ############################################  
  
  #######################################################################
  ##                          identify best model                      ##
  ## (set it to bestmodel, and set best hyper parameters if available) ##
  #######################################################################  
  
      #now that we've fit all the models, pick the best one.
  bestmodel.in = order(allmodelCV.in)[1]

  ### finally, fit the best model to the full (available. i.e. outer CV's training set) data. 
  #if the model was a penalized regression, then use the tuned hyper parameters as well.
  if (bestmodel.in <= nLinmodels) {  # then best is one of linear models
    bestfit = lm(formula = allLinModels[[bestmodel.in]],data=fulldata.in)  # fit on all available data
    bestcoef = coef(bestfit)
  } else if (bestmodel.in <= nRRmodels+nLinmodels) {  # then best is one of RR models
    bestlambdaRR = (lambdalistRR)[bestmodel.in-nLinmodels]
    bestfit = glmnet(x.in, y.in, alpha = 0,lambda=lambdalistRR)  # fit the model across possible lambda
    bestcoef = coef(bestfit, s = bestlambdaRR) # coefficients for the best model fit
  } else {  # then best is one of LASSO models
    bestlambdaLASSO = (lambdalistLASSO)[bestmodel.in-nLinmodels-nRRmodels]
    bestfit = glmnet(x.in, y.in, alpha = 1,lambda=lambdalistLASSO)  # fit the model across possible lambda
    bestcoef = coef(bestfit, s = bestlambdaLASSO) # coefficients for the best model fit
  }

  #########################################
  ##          Making predictions         ##
  ## (using best model/hyper parameters) ##
  #########################################  
   
    #using the empty vector, assign this iterations best model number so we can keep track of it.
  allbestmodels[j] = bestmodel.in
  
    #finally, perform the prediction process on this iterations "test" set using this iterations "best" model.
    #so for data points with indices of 1, use model 6 (which is a linear regression).
  if (bestmodel.in <= nLinmodels) {  # then best is one of linear models
    allpredictedCV.out[groupj.out] = predict(bestfit,validdata.out)
  } else if (bestmodel.in <= nRRmodels+nLinmodels) {  # then best is one of RR models
    allpredictedCV.out[groupj.out] = predict(bestfit,newx=validdata.out,s=bestlambdaRR)
  } else {  # then best is one of LASSO models
    allpredictedCV.out[groupj.out] = predict(bestfit,newx=validdata.out,s=bestlambdaLASSO)
  }
  
  ####################################
  ##   End of CV outer loop         ##
  ## (will recycle for k.out times) ##
  ####################################  
  
}

#############################################################
##### Now 2x CV is done, lets assess the results        #####				 
#############################################################

# for curiosity, we can see the models that were "best" on each of the inner splits
allbestmodels

#assessment
  #pull all of the y's out of the original dataset
y.out = fulldata.out$BodyFatSiri
  #calculate MSE of all the predicted values - actual values
CV.out = sum((allpredictedCV.out-y.out)^2)/n.out; CV.out
  #calculate R2 value based on all these predictions. 
R2.out = 1-sum((allpredictedCV.out-y.out)^2)/sum((y.out-mean(y.out))^2); R2.out

