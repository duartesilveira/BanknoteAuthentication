import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB as gb
import math 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 

#-----------------------------FUNCTIONS--------------------------------------------------------------
#WE BEGIN BY CREATING FUNCTIONS THAT HELP US TO ORGANIZE THE CODE AND LATER COMPUTE THE DESIRED RESULTS


#-----------------------------GENERAL FUNCTIONS------------------------------------------------------

# Function that loads the data
# The input is the file name
# Returns a numpy matrix with the data.
def load_data(file_name):
    matrix = np.loadtxt(file_name, delimiter='\t') 
    return matrix

# Function that returns stratified-folds for n fold cross-validation.
# The input is the number of folds.
# Returns the n stratified folds.
def get_k_folds (n_folds):
    kf = StratifiedKFold(n_splits=n_folds)   
    return kf

#--------------------------------KDE NAIVE BAYES-----------------------------------------
   

# Auxiliar function that completely determines the estimated joint probability distribution (without prior probabilities).
# Inputs:
# -Bandwidth parameter for Kernel Density estimator.
# -Training data to fit the distributions.
# Returns:
# -Prior probabilities of each class.
# -Kernel Density estimated distributions for each feature-class combination, fitted on the training data.

def kde_naive_bayes(X_train,Y_train,bw):  
    #divide the training set into two matrixes, one for each class
    matrix_0=[]
    matrix_1=[]
    for i in range(len(Y_train)):
        if Y_train[i]==0:
            matrix_0.append(X_train[i])
        else: 
            matrix_1.append(X_train[i])
    #convert the matrixes into numpy arrays
    matrix_0=np.array(matrix_0)
    matrix_1=np.array(matrix_1)
    #prior probabilities for each class
    prior_prob_0 = len(matrix_0)/len(X_train)
    prior_prob_1 = len(matrix_1)/len(X_train)
    #vectors to store the conditional distributions on each class
    kde_0=[]
    kde_1=[]
    #kernel estimator distribution for each feature-class combination
    for i in range(0,4):   #last is not feature
        #create KernelDensity object, fit with training data and store the distributions
        kde_0_k = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde_0_k.fit(matrix_0[:,i].reshape(-1, 1))
        kde_0.append(kde_0_k)
        #create KernelDensity object, fit with training data and store the distributions
        kde_1_k = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde_1_k.fit(matrix_1[:,i].reshape(-1, 1))
        kde_1.append(kde_1_k)
    #convert into numpy arrays 
    kde_0=np.array(kde_0)
    kde_1=np.array(kde_1)
    
    return (prior_prob_0,prior_prob_1,kde_0,kde_1)

#--------------------------------------------------------------------------------------------------------

# Function that predicts the class values using KDE Naive Bayes. 
# Uses the estimated joint probability functions to predict the class, given some features.
# Inputs:
# -The prior probabilities of each class.
# -The estimated kernel density distributions, fitted with training data.
# -A matrix X with the features used to predict the classes.
# Returns:
# -A binary array with the predicted class values.
def kde_naive_bayes_predict(X, prior_prob_0, prior_prob_1, kde_0,kde_1):  
    
    score_samples_0=[]
    score_samples_1=[]
    #computes de score samples for each feature-class combination.
    #the outputs are two matrixes, with the log-likelihoods for each class, given the input data. 
    for i in range(0,4):
        score_samples_0.append(kde_0[i].score_samples(X[:,i].reshape(-1, 1)))
        score_samples_1.append(kde_1[i].score_samples(X[:,i].reshape(-1, 1)))
        
    #vector to store the predictions
    predictions = []
    
    #for each point of X does the sum of the features log-conditional probs and the log-prior prob of each class
    for i in range(len(X)):
        sum_features_0=0
        sum_features_1=0
        for j in range(0,4):
            sum_features_0=sum_features_0+score_samples_0[j][i]
            sum_features_1=sum_features_1+score_samples_1[j][i]
            
        p0= math.log(prior_prob_0)+ sum_features_0
        p1= math.log(prior_prob_1)+ sum_features_1
        #chooses class by comparing and choosing the highest probability
        if p0>p1:
            predictions.append(0)
        else:
            predictions.append(1)
    
    return predictions

#-----------------------------------------------------------------------------------

# Function that measures the performance of KDE Naive Bayes classifier
# Inputs:
# -Binary array with the predictions when used KDE Naive Bayes
# -The actual observed class values one wants to predict.
# Returns:
# -the fraction of incorrect classifications in the test set using KDE Naive Bayes
# -a binary array with 1 if correctly predicted and 0 otherwise
def kde_naive_bayes_score(predictions,Y_test):
    #accuracy gives the fraction of correct predictions
    accuracy=accuracy_score(Y_test, predictions)
    #error gives the fraction of incorrect predictions
    test_error=1-accuracy    
    #create a binary array of same size that tells if got right or not for each prediction
    times_right_wrong = (predictions == Y_test )*1
          
    return (test_error, times_right_wrong)

#-----------------------------------------------------------------------------------

# Function that performs cross-validation in order to optimize the bandwidth parameter.
# Inputs:
# -The training data, which will be sucessively splitted in train and validation sets, during this process.
# -StratifiedKFold object, kf, that sets the number of folds in cross-validation.
# -The prior probabilities of each class.
# Returns:
# -The optimal bandwidth.
# -Vector with the average training error for each bandwidth, used to perform the plot
# -Vector with the average cross-validation error for each bandwidth, used to perform the plot
# -Vector with all the tried bandwidths, used to perform the plot
def kde_nb_cross_valid_bandwidth(X,Y,kf):
    
    # Vectors to store the training and test errors
    training_errors=[]
    validation_errors=[]
    
    # Try values for the bandwidth
    bandwidths=np.arange(0.02,0.62,0.02)
    
    #for each value of bandwidth we want to test
    for bw in bandwidths:    
        sum_trai_err=0
        sum_val_err=0
                
        #we fit the model to the training and then get the predicitons for both the training and validation
        #to be able to get the average training error and validation error of each group
        for train,valid in kf.split(Y,Y):
            [prior_0,prior_1,kde_0, kde_1]=kde_naive_bayes(X[train],Y[train],bw)
            predictions_train=kde_naive_bayes_predict(X[train,:],prior_0, prior_1, kde_0, kde_1)
            predictions_val=kde_naive_bayes_predict(X[valid,:],prior_0, prior_1, kde_0, kde_1)
            sum_trai_err+= kde_naive_bayes_score(predictions_train,Y[train])[0]
            sum_val_err+= kde_naive_bayes_score(predictions_val,Y[valid])[0]
        
        #append in the arrays the average of the training and validation error for the 5 groups
        training_errors.append(sum_trai_err/kf.n_splits)
        validation_errors.append(sum_val_err/kf.n_splits)
        
    #bandwidth for which validation error is minimum
    bw_low=bandwidths[validation_errors.index(min(validation_errors))]
    
    return (bw_low,training_errors,validation_errors,bandwidths)


#------------------LOGISTIC REGRESSION---------------------------------------------------

# Auxiliar function that measures Brier Score of training and validation sets in cross-validation
# Brier Score was used instead of the fraction of incorrect classifications because its plots are more interesting
# Inputs:
# -Training data, on which cross-validation is performed.
# -Training and validation indexes, obtained when dividing the data set into
# -The regularization parameter C.
# Returns:
# -Brier Score of the training set
# -Brier Score of the validation Set
def LR_Brier_Score_Cross_Val( X,Y, train_ix,val_ix,C_value):
    #create a LogisticRegression object
    reg=LogisticRegression(penalty='l2',C=C_value, tol=1e-10)
    #fit with the training set
    reg.fit(X[train_ix,:],Y[train_ix])
    #vector with the squared difference between the predicted probabilities and the real class values
    prob = reg.predict_proba(X[:,:])[:,1]
    squares = (prob-Y)**2
    #return training and validation brier score
    return (np.mean(squares[train_ix]),np.mean(squares[val_ix]))

#--------------------------------------------------------------------------    

# Function that performs cross-validation in order to optimize the regularization parameter C
# Brier Score was used instead of the fraction of incorrect classifications because its plots are more interesting
# Inputs:
# -The training data, which will be sucessively splitted in train and validation sets, during this process.
# -StratifiedKFold object, kf, that sets the number of folds in cross-validation.
# Returns:
# -log(optimal C), the base 10 logarithm of the optimal C parameter
# -vector with the average training error for each C, used to perform the plot
# -vector with the average cross-validation error for each C, used to perform the plot
# -vector with all the tried log(C) paramenters, used to perform the plot.
def LogRegression_cross_valid_C (X,Y, kf):
    #vectors to store the training and test errors
    training_errors=[]
    validation_errors=[]    
    #try values for log(C) parameter
    logc_params=np.arange(-2,13)    
    #for every log(C) in try values, compute the average training a validation errors, using Brier Score
    for logc in logc_params:
        sum_trai_err=0
        sum_val_err=0 
        #sum the training a validation error through all the 5 folds
        for train,valid in kf.split(Y,Y):
            sum_trai_err+= LR_Brier_Score_Cross_Val( X,Y, train,valid,float(10)**logc)[0]
            sum_val_err+=  LR_Brier_Score_Cross_Val( X,Y, train,valid,float(10)**logc)[1]   
        #append in the arrays the average of the training and validation error for the 5 groups
        training_errors.append(sum_trai_err/kf.n_splits)
        validation_errors.append(sum_val_err/kf.n_splits)
    #log(C) for which validation error is minimum
    logC_low=logc_params[validation_errors.index(min(validation_errors))]
        
    return (logC_low,training_errors,validation_errors,logc_params) 

#-----------------------------------------------------------------------------
     
# Function that measures the performance of LogisticRegression classifier. 
# When this performance is measured on the test set we are estimating true error
# Inputs:
# -The training data, used to train the model
# -The regularization parameter C, used in Logistic Regression. To estimate true error we use optimal C
# -The test data, on which we measure prediction accuracy (1-accuracy)
# Returns:
# -the fraction of incorrect classifications in the test set using LogisticRegression
# -a binary array with 1 if correctly predicted and 0 otherwise
def LogRegression_score( X_train,Y_train, X_test, Y_test, C_value):    
    #create a LogisticRegression object
    reg=LogisticRegression(penalty='l2',C=C_value, tol=1e-10)
    #fit with the training set
    reg.fit(X_train,Y_train)
    #get test error (fraction of incorrect classifications)
    test_error=1-reg.score(X_test[:,:],Y_test)
    #create a binary array of same size that tells if got right or not for each prediction
    predictions= reg.predict(X_test)
    vector_right_wrong = (predictions == Y_test )*1   
    
    return (test_error, vector_right_wrong)

#-------------------GAUSSIAN NAIVE BAYES------------------------------------

# Function that measures the performance of GaussianNB classifier
# Inputs:
# -The training data, used to train the model
# -The test data, on which we measure prediction accuracy (1-accuracy)
# Returns:
# -The fraction of incorrect classifications in the test set using GaussianNB
# -A binary array with 1 if correctly predicted and 0 otherwise
def gaussian_nb_score(X_train,Y_train,X_test,Y_test):   
    #create GaussianNB object
    model= gb()
    #fit with the training set
    model.fit( X_train, Y_train.ravel()) 
    #get the accuracy
    accuracy=model.score(X_test,Y_test) 
    #create a binary array of X_test size that tells if predicted right or not for each prediction
    predictions=model.predict(X_test)  
    vector_right_wrong = (predictions == Y_test )*1
    #get test error
    test_error= 1-accuracy    
    return (test_error, vector_right_wrong)
    
#------------------MODEL COMPARISION---------------------------


# Function that performs McNemars test between two models/classifiers
# Inputs:
# -Binary arrays returned on the score functions (1 if correctly predicted and 0 otherwise), for both models in comparision.
# Returns:
# -The number of times model 1 prediction was correct and model 2 prediction as not (e10)
# -The number of times model 2 prediction was correct and model 1 prediction as not (e01)
# -The value of McNemars chi-squared statistic
def mcnemars_test(right_wrong_model1, right_wrong_model2):
    #count the number of times each model got right and the other not
    e01=0
    e10=0
    for i in range(len(right_wrong_model1)):
        if right_wrong_model1[i]>right_wrong_model2[i]:
            e01+=1
        elif right_wrong_model1[i]<right_wrong_model2[i]:
            e10+=1            
    #if the models have exactly predictions, the statistic is zero
    if e01+e10==0:
        test=0
    #otherwise, use the formula in the lecture notes
    else:
        test=((abs(e01-e10)-1)**2)/(e01+e10)
        
    return (e01,e10,test)

#------------------------------------------------------------------------

# Function that computes approximate normal test interval for each model/classifier
# -Binary array returned on the score function (1 if correctly predicted and 0 otherwise), for the model in consideration.
# Returns:
# -The 95% confidence interval for the expected number of errors
def aproximate_normal_test(right_wrong_model):
    #get the observed number of errors
    X=0
    for i in range(len(right_wrong_model)):
        if (right_wrong_model[i]==0):
            X=X+1
    #estimate the probability of error
    p0=X/len(right_wrong_model)
    #compute the binomial standard deviaton
    st_dev= math.sqrt(len(right_wrong_model)*p0*(1-p0))
    #return the confidence interval
    return (X-1.96*st_dev,X+1.96*st_dev)



#-------------------------RESULTS--------------------------------------------------------------------------------
# NOW WE USE THE PREVIOUS FUNCTIONS TO PRESENT THE RESULTS AND DRAW CONCLUSIONS

#LOAD THE TRAINING DATA
train_data= load_data('TP1_train.tsv')
#RANDOMIZE THE ORDER OF THE DATA POINTS
train_data=shuffle(train_data)
#DIVIDE THE TRAINING DATA INTO FEATURES AND CLASSES
Y_train=train_data[:,4]
X_train=train_data[:,:4]
#STANDARDIZE THE ATTRIBUTE VALUES
means=np.mean(X_train, axis=0)
stdevs=np.std(X_train,axis=0)
X_train=(X_train-means)/stdevs

#LOAD THE TEST DATA
test_data= load_data('TP1_test.tsv')
#DIVIDE THE TEST DATA INTO FEATURES AND CLASSES
Y_test=test_data[:,4]
X_test=test_data[:,:4]
#STANDARDIZE THE ATTRIBUTE VALUES
X_test=(X_test-means)/stdevs


#GET THE 5 FOLDS TO PERFORM CROSS-VALIDATION
kf= get_k_folds (5)


#---------------------------LOGISTIC REGRESSION-------------------------------------------
    
#OPTIMIZE THE PARAMETER C WITH CROSS-VALIDATION FOR LOGISTIC REGRESSION.
[logc_optimal,training_errors,validation_errors,logC_params]= LogRegression_cross_valid_C (X_train,Y_train,kf)

#PRINT LOG10(OPTIMAL C).
print("optimal log(c):" + str(logc_optimal))

#PLOT TRAINING AND CROSS-VALIDATION ERRORS FOR THE TRIED C PARAMETERS
plt.figure(figsize=(8, 6))
plt.plot(logC_params,training_errors,color="red", label="Training Error", linewidth=1.0)
plt.plot(logC_params,validation_errors, color="black", label="Validation Error", linewidth=1.0)
plt.title(f"Optimal C for Logistic Regression: {logc_optimal}")
plt.legend(['Training Error', 'Validation Error'])
plt.xlabel('log(C) value')
plt.ylabel('Error')
plt.scatter(
logC_params[validation_errors.index(min(validation_errors))],
min(validation_errors),
label="Optimal C value",
s=100,
c="black",
)
plt.xlim(-2, 12)
plt.legend()
plt.savefig('LR.png') 
plt.show()   
plt.close()

# --------------------------WITH C OPTIMIZED----------------------------

#COMPUTE THE TEST ERROR
#COMPUTE BINARY VECTOR THAT TELLS IF THE PREDICTION USING LR WAS ACURATE (1) OR NOT (0), REGARDING THE TEST SET.
[LR_test_error,LR_binary]=LogRegression_score( X_train,Y_train, X_test, Y_test, float(10)**logc_optimal)

#------------------------- KDE NAIVE BAYES CLASSIFIER------------------------------------------

#OPTIMIZE THE BANDWIDTH PARAMETER USING CROSS- VALIDATION FOR KDE
[bw_opt,training_errors,validation_errors,bandwidths]= kde_nb_cross_valid_bandwidth (X_train,Y_train, kf)

#PRINT OPTIMAL BANDWIDTH
print("optimal bandwidth:" + str(bw_opt))

#PLOT OF TRAINING AND CROSS-VALIDATION ERRORS FOR THE TRIED BANDWIDTH PARAMETERS
plt.figure(figsize=(8, 6))
plt.plot( bandwidths,training_errors,color="red", label="Training Error", linewidth=1.0)
plt.plot( bandwidths,validation_errors, color="black", label="Validation Error", linewidth=1.0)
plt.title(f"Optimal Bandwidth for Naive Bayes: {round(bw_opt,2)}")
plt.legend(['Training Error', 'Validation Error'])
plt.xlabel('Bandwidth')
plt.ylabel('Error')
plt.scatter(
bandwidths[validation_errors.index(min(validation_errors))],
min(validation_errors),
label="Optimal Bandwidth",
s=100,
c="black",
)
plt.xlim(0.02, 0.6)
plt.legend()
plt.savefig('NB.png')  
plt.show()  
plt.close()

# --------------------------WITH BANDWIDTH OPTIMIZED----------------------------

#FIT WITH TRAINING DATA
[prior_0,prior_1 ,kde_0, kde_1]=kde_naive_bayes(X_train,Y_train, bw_opt)

#PREDICT USING TEST DATA
predictions=kde_naive_bayes_predict(X_test,prior_0, prior_1, kde_0, kde_1)

#COMPUTE THE TEST ERROR
#COMPUTE BINARY VECTOR THAT TELLS IF THE PREDICTION USING LR WAS ACURATE (1) OR NOT (0), REGARDING THE TEST SET
[KDE_test_error,KDE_binary]=kde_naive_bayes_score(predictions,Y_test)


#--------------------------- GAUSSIAN NAIVE BAYES----------------------------------

#COMPUTE THE TEST ERROR
#COMPUTE BINARY VECTOR THAT TELLS IF THE PREDICTION USING LR WAS ACURATE (1) OR NOT (0), REGARDING THE TEST SET
[GAUSS_test_error,GAUSS_binary]= gaussian_nb_score(X_train,Y_train,X_test,Y_test)


#----------------------------COMPARING MODELS----------------------------------------------


#----------------------------TEST ERRORS---------------------------------------------------
print("")
print("TEST ERRORS ")
print("")
print("LR: "+str(LR_test_error)+ "  KDE: "+str(KDE_test_error)+ "  GAUSSIAN: " +str(GAUSS_test_error ))


#----------------------------MCNEMARS TEST--------------------------------------------------
print("")
print("Mcnemars Test: (if value<=3.84, the classifiers perform identically)")

#NAIVE BAYES WITH KDE VS GAUSSIAN NAIVE BAYES
[e01,e10,test]=mcnemars_test(KDE_binary, GAUSS_binary)
print("")
print("Naive Bayes with KDE vs Gaussian Naive Bayes: ")
print("")
print("e01: "+str(e01)+ " e10: "+str(e10)+ " test: "+str(test))

#NAIVE BAYES WITH KDE VS LOGISTIC REGRESSION
[e01,e10,test]=mcnemars_test(KDE_binary, LR_binary)
print("")
print("Naive Bayes with KDE vs LR: ")
print("")
print("e01: "+str(e01)+ " e10: "+str(e10)+ " test: "+str(test))

#GAUSSIAN NAIVE BAYES VS LOGISTIC REGRESSION
[e01,e10,test]=mcnemars_test(GAUSS_binary, LR_binary)
print("")
print("Gaussian Naive Bayes vs LR: ")
print("")
print("e01: "+str(e01)+ " e10: "+str(e10)+ " test: "+str(test))

#---------------------------- APROXIMATE NORMAL TEST-------------------------------
print("")
print("APROXIMATE NORMAL TEST")

#LOGISTIC REGRESSION
print("")
print("Logistic Regression")
print("")
print("interval: "+ str(aproximate_normal_test(LR_binary)))

#NAIVE BAYES WITH KDE
print("")
print("KDE Naive Bayes")
print("")
print("interval: "+ str(aproximate_normal_test(KDE_binary)))

#GAUSSIAN NAIVE BAYES 
print("")
print("GAUSSIAN Naive Bayes")
print("")
print("interval: "+ str(aproximate_normal_test(GAUSS_binary)))


 