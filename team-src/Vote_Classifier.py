from Imports import *

#------------------------------------------------Vote Classifier Functions-------------------------------------------#
#These are functions made to combine all previous models and see if that will return more accurate results  
#The Built in function was giving us trouble with the ANN so we decided to make the functions by hand. It has the same structure but cannot be trained like the previous one I beleive 
# Authors: Lucas Dillistone 
#------------------------------------------------------------------------------------------------------------#



#The purpose of this function is to predict the answers given a model and some data 
#it takes two inputs one that is the model and one that is the data to predict on 
def OutputPredictions(model, Xdata):
    predictions=model.predict(Xdata)
    predictions=np.round(predictions,0)
    predictions=predictions.astype(int)
    return predictions


#The purpose of this function is to find the accuracy of the combined models
#It will take two inputs testin for the produced results and testout for the results to check against 
#there will be no output 
def Vote_acc (testin,testout):
    testout=testout.values
    count=0#set the count for each varible to zero for later 
    count1=0
    count0=0
    for i in range(0,(testin.size)):#Go through all of the values and see if it matched for the overall accuracy, see if both are 1 for accuracy of 1 and check if both are 0 for accuracy of 0
        if(testin[i]==testout[i]):
            count+=1
        if( testin[i]==1 and testout[i]==1):
            count1+=1
        if( testin[i]==0 and testout[i]==0):
            count0+=1 
    acc=((count/testout.size))*100
    acc1=(count1/np.count_nonzero(testout==1))*100
    acc0=(count0/np.count_nonzero(testout==0))*100
    print("Total accuracy: %d    Accuracy of predicting fraud: %d     Accuracy of predicting non-fraud: %d"%(round(acc),round(acc1),round(acc0)))#print the accuracy of each variable
    print("AUC %.4f"%(roc_auc_score(testout,testin)))#ouput the ROC curve 

    #plot the roc curve
    fprK, tprK, threshK=roc_curve(testout,testin)#get the ROC curve 
    aucK=auc(fprK,tprK)#find the AUC value
    plt.figure(1)#create and ouput the plot 
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fprK, tprK, label=' area = {:.3f}'.format(aucK))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


#the purpose of this function is to combine all of the models created before to find the most likely outcome based on all of the predictions 
#it will take two inputs models which is a list of the model names and models and train_x which is the data which will be used and test_y which is what it will be checked agaisnt 
#It will return the predicted outcomes of the data after combination 
def Hard_Vote_Classifier(models,train_y,test_y):
    predictions=list()
    CombPred=[]
    for i in range(0,len(models)):
        pred=OutputPredictions(models[i],train_y)
        predictions.append(pred)
    for m in range(0,len(predictions[0])):
        sumval=0
        for l in range(0,len(models)):
            sumval=sumval+predictions[l][m]
        sumval=sumval/len(models)
        sumval=np.round(sumval,0)
        sumval=sumval.astype(int)
        CombPred.append(sumval)
    CombPred=np.array(CombPred)
    Vote_acc(CombPred,test_y)
    return CombPred
