#import all of the files and functions 
from Imports import * 
from ANN import *
from DataPreprocessing import *
from EvaluateModel import *
from FeatureEngineering import *
from HandleImbalancedDataset import *
from Metrics import *
from PlotROC import *
from ReadDataset import *
from TrainTestSplit import *
from XGBoostModel import * 
from Vote_Classifier import *


com_dat=readMergedDatasetLocal()#get the data 


#Combined_Data=readMergedDataset()#get the data if using drive
com_dat=reduceMemUsage(com_dat)#Reduce the memory usage 
#start of the data processing 
train_features, train_labels=prepare_inputs_and_outputs(com_dat)

#find and drop missing data 
missingData=get_missing_data_percentage(train_features)#find values with a lot of missing data 
train_features = drop_high_missing_data_columns(missingData, train_features, 70)

#drop columns with high correlation or with only one value
train_features=drop_one_value_columns(train_features)#drop columns with only one value
train_features = drop_high_correlation_features(train_features, 0.80)

#encode the data so categorical turns into numeric 
train_features=encodeNumericalColumns(train_features)

train_features=replaceMissingValues(train_features)#replace all of the missing values
 
 #start of the data transformation 
X_train, Y_train, X_test, Y_test, X_validation, Y_validation = split_data(train_features, train_labels)

#Select the 50 best features 
X_train, X_validation, X_test = selectkbestfeatures(X_train, Y_train, X_validation, X_test, 50)

 # Feature Scaling using Standardization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)

#Data Reduction using PCA 
pca = PCA(n_components=25).fit(X_train)

X_train_pca = pca.transform(X_train)
X_validation_pca = pca.transform(X_validation)
X_test_pca = pca.transform(X_test)

X_train = pd.DataFrame(data = X_train_pca)
X_validation = pd.DataFrame(data = X_validation_pca)
X_test = pd.DataFrame(data = X_test_pca)

#start to deal with the imbalanced data 
X_sampled, Y_sampled= handleImbalancedDataset("smotetomek",X_train,Y_train)

#Building the ANN Model
nn_mod,nnHist=NNetwork(X_sampled,Y_sampled,X_validation,Y_validation,10,300,700,25,400,0)#Set it to work with the combined data testing on the validation with 5 epochs and the layers specified 
NNacc(nn_mod,X_test, Y_test)#find the accuracy of the model on the testing data 
PlotHistory(nnHist)#plot the historical loss and accuracy over the epochs 
plot_model(nn_mod,to_file="ANN_Model_Structure.png", show_shapes=True, show_layer_names=True)#show the structure of the ANN


#Building the XGBoost Model 
XGB_mod=XGModel(X_sampled,Y_sampled,X_validation,Y_validation)#create the model 
XGAccuracy(XGB_mod,X_test,Y_test)#Find the accuracy and plot the ROC
XGFeatureImportance(XGB_mod)#Get the feature importances 
#XGTree(XGB_mod) #Plot a tree(this is a very large tree and is not viewable unless the maximum depth is very shallow)
XG_cv=XGCV(X_sampled,Y_sampled)#perform some cross validation 

#DAVID AND KEVIN PUT YOUR MODELS HERE THEN ADD THEM TO THE COMBINATION THING 


#Use the voting classifier to combine the models 
models=list()#Create a list of the models 
models.append(nn_mod)#add a model 
models.append(XGB_mod)#add a model 
#add more models as needed
print("Results of Voting classifier on sampled data ")
together=Hard_Vote_Classifier(models,X_sampled,Y_sampled)#test on sampled data 
print("Results of Voting classifier on testing data ")
test=Hard_Vote_Classifier(models,X_test,Y_test)#test on the testing data 
print("Results of Voting classifier on validation data ")
validation=Hard_Vote_Classifier(models,X_validation,Y_validation)#test on the validation data 
