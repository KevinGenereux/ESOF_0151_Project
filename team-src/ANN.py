from Imports import * 
#------------------------------------------------Neural Network Functions-------------------------------------#
#These are different functions to build and test out an Artifical Neural Network 
#There are functions to train the model, print training data, find accuaracy and other metrics and plot the layout of the NN
# Authors: Lucas Dillistone 
#------------------------------------------------------------------------------------------------------------#

#the NNetwork function will build and train a Neural Network 
#there will be 8 inputs but not all of them will be used. Tr is the predictor data, A is the response data E is the epochs L1-L4 are the different layer sizes but some may not be in use and W is a binary 0 or 1 to see if custom weights will need to be given to the answers 
#the output of this model is the NNModel that has been trained and the fitted model from the data 
def NNetwork(Tr,A,val_x,val_y,E,L1,L2,L3,L4,W):#Tr is training set A is answer L1 is layer 1 nodes L2 is layer 2 nodes E is the epochs and W is if weights are needed
    NNIn=Tr#take the data and use  new name for it 
    NNans=A
    ##########################################layers are added and removed here to test different implementations
    model_NN=Sequential()
    model_NN.add(Dense(L1, input_dim=NNIn.shape[1], activation='relu'))#50 nodes in first hidden layer there might have been an issue with relu but it seems to be fixed
    model_NN.add(Dropout(0.1))
    model_NN.add(Dense(L2, activation='relu'))
    #model_NN.add(Dense(L3, activation='relu',kernel_initializer='uniform'))
    
    #model_NN.add(Dense(L4, activation='relu'))#seems to bring back the all same value issue 
    #model_NN.add(Dropout(0.1))
    model_NN.add(Dense(1, activation='sigmoid'))
    model_NN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model_NN.compile(optimizer=sgd,loss='mse')##################################################try another optimizer
    if(W==1):#if the weights are needed or not 
        weights={0:(np.count_nonzero(NNans==1)/NNans.size), 1:(np.count_nonzero(NNans==0)/NNans.size)}#gives more even distribution less loss and not much difference in accuracy 
        #weights={0:7, 1:193}#increase to 27 gave slightly better results 
        nnHist=model_NN.fit(NNIn,NNans,class_weight=weights,epochs=E, validation_data=(val_x,val_y))# it says 11073 insteead of 354324 because there are 32 batches and 354324/32=11073
        
    else: 
       nnHist=model_NN.fit(NNIn,NNans,epochs=E, validation_data=(val_x,val_y))
    
   
    return model_NN, nnHist
    NNacc(model_NN,NNIn,NNans)#call the other function to analyze 


#this function can be called to display the accuracy and outputs of a model. The ROC curve will also be printed
#it will take three inputs model which is the trained model, testin which is the predictor values and testout which is the response values 
#there will be no return as the point is to output some results and plot the ROC curve 
def NNacc(model, testin,testout):
    predictions=model.predict_classes(testin)#make predictions 
    check=np.array(testout)#convert to an array 
    count=0#set the count for each varible to zero for later 
    count1=0
    count0=0
    for i in range(predictions.size):#Go through all of the values and see if it matched for the overall accuracy, see if both are 1 for accuracy of 1 and check if both are 0 for accuracy of 0
        if(predictions[i]==check[i]):
            count+=1
        if( predictions[i]==1 and check[i]==1):
            count1+=1
        if( predictions[i]==0 and check[i]==0):
            count0+=1 
    acc=((count/testout.size))*100
    acc1=(count1/np.count_nonzero(check==1))*100
    acc0=(count0/np.count_nonzero(check==0))*100
    print("Total accuracy: %d    Accuracy of predicting fraud: %d     Accuracy of predicting non-fraud: %d"%(round(acc),round(acc1),round(acc0)))#print the accuracy of each variable
    print("AUC %.4f"%(roc_auc_score(testout,predictions)))#ouput the ROC curve 

    #plot the roc curve
    fprK, tprK, threshK=roc_curve(testout,predictions)#get the ROC curve 
    aucK=auc(fprK,tprK)#find the AUC value
    plt.figure(1)#create and ouput the plot 
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fprK, tprK, label='Keras (area = {:.3f})'.format(aucK))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    

#This function will plot the accuracy and loss of the model along with the validation for each value 
#it will take one value as input nnHist which is the history of fitting the model 
#there will be no return because the point is to just output some
def PlotHistory(nnHist):
    print(nnHist.history.keys())
    plt.plot(nnHist.history['accuracy'])
    plt.plot(nnHist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(nnHist.history['loss'])
    plt.plot(nnHist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

