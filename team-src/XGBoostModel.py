from Imports import *

#------------------------------------------------XGBoost Functions-------------------------------------------#
#These are functions made to use the XGBoost model to train and test the data 
#Additional functions are made to plot the data and find the results  
# Authors: Lucas Dillistone 
#------------------------------------------------------------------------------------------------------------#

#Function to build the xgmodel with the parameters specified. Different parts may be commented and uncommented based on what you are trying to build 
#will take 4 inputs predictors for the predictor data, response for the response data, test_pred for the validation prediction data and test_resp for the validation response data
#this function will only return the trained model 
def XGModel(predictors,response,test_pred,test_resp):
       # xg_reg=xgb.XGBRegressor(objective='reg:logistic',colsample_bytree=0.15, learning_rate=0.01,max_depth=12,n_estimators=2000,subsample=0.4 ,tree_method='gpu_hist',                       eval_metric='auc',alpha=10)#create the regressor model with specific values 

        
        xg_reg = xgb.XGBClassifier(  # a classifyer XGBoost model with specific values to be used later 
                n_estimators=2000,
                max_depth=6, 
                objective='binary:logistic',
                learning_rate=0.3, 
                subsample=0.7,
                colsample_bytree=0.15, 
                missing=-1, 
                eval_metric='auc',
                scale_pos_weight=2,
                # USE CPU
                nthread=4,
                tree_method='hist' 
                # USE GPU
                #tree_method='gpu_hist' 
                )
         


        xg_reg.fit(predictors, response, eval_set=[(test_pred,test_resp)],verbose=50, early_stopping_rounds=100)#fit thr XGBoost model with the data we are looking for along with a validation set it will print the result of every 50 trials and will stop if there has been no improvement after 100 trials 

        return xg_reg

#this function will display the auc and RMSE scores of a model that has been trained on data we have specified it will then plot the AUC curve 
# it will take three inputs Model which is the model that was trained, X_dat which is the predictor data and Y_dat which is the response data 
#there will be no returns as the function will be printing 
def XGAccuracy(Model,X_dat,Y_dat):
        pred=Model.predict(X_dat)
        pred=[round(value) for value in pred]
        rmse=np.sqrt(mean_squared_error(Y_dat,pred))
        print("RMSE: %f"%(rmse))
        print("AUC: %.4f"%(roc_auc_score(Y_dat,pred)))


        fpr, tpr, _=roc_curve(Y_dat, pred)
        roc_auc=auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()


#this function will create two plots showing the feature importance of the features used in the model 
#it will take one input variable with the model 
#there will be no return as the objective is to print the plots 
def XGFeatureImportance(mod):
        
        #fig1=pyplot.bar(range(len(mod.feature_importances_)), mod.feature_importances_)
        #pyplot.show()
        #fig1.savefig("..\Plots\Feature_Importance1.png")
        xgb.plot_importance(mod)
        pyplot.show()




#This function will print out a tree from the XGBoost model but unless the depth is very low it will not be enough to display 
#this will only take one input which is the model 
#there will be no return as it is just attempting to print the tree
def XGTree(mod):
        fig, ax=plt.subplots(figsize=(30,30))
        xgb.plot_tree(mod, num_trees=4, ax=ax)
        plt.show()
        

#this function will perform some cross validation and comperisons on the XGBoost Data 
#it will take in two variables x_dat which is the predictor values and y_dat which is the response values 
#it will return the results of the cross validation 
def XGCV(x_dat,y_dat):
        data_dmatrix=xgb.DMatrix(data=x_dat,label=y_dat)
        #attempting to find results with k-fold cross validation 

        params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.01,'max_depth': 22, 'alpha': 10}
        cv_results = xgb.cv(dtrain=data_dmatrix ,params=params, nfold=10,num_boost_round=10,early_stopping_rounds=10,metrics="auc", as_pandas=True ,seed=123)
        print(cv_results)

        return cv_results

