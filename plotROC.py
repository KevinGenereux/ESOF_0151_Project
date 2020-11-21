from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, auc, roc_curve

def plot_roc_curve(X_train, Y_train, X_validation, Y_validation, X_test, Y_test):

  # predict probabilities
  adaboost_probs_train = cv.predict_proba(X_train)
  adaboost_probs_validation = cv.predict_proba(X_validation)
  adaboost_probs_test = cv.predict_proba(X_test)

  # keep probabilities for the positive outcome only
  adaboost_probs_train = adaboost_probs_train[:, 1]
  adaboost_probs_validation = adaboost_probs_validation[:, 1]
  adaboost_probs_test = adaboost_probs_test[:, 1]

  # calculate scores
  adaboost_auc_train = roc_auc_score(Y_train, adaboost_probs_train)
  adaboost_auc_validation = roc_auc_score(Y_validation, adaboost_probs_validation)
  adaboost_auc_test = roc_auc_score(Y_test, adaboost_probs_test)

  # summarize scores
  print('Train Set: ROC AUC=%.3f' % (adaboost_auc_train))
  print('Validation Set: ROC AUC=%.3f' % (adaboost_auc_validation))
  print('Test Set: ROC AUC=%.3f' % (adaboost_auc_test))

  # calculate roc curves
  adaboost_fpr_train, adaboost_tpr_train, _ = roc_curve(Y_train, adaboost_probs_train)
  adaboost_fpr_validation, adaboost_tpr_validation, _ = roc_curve(Y_validation, adaboost_probs_validation)
  adaboost_fpr_test, adaboost_tpr_test, _ = roc_curve(Y_test, adaboost_probs_test)

  # plot the roc curve for the model
  pyplot.plot(adaboost_fpr_train, adaboost_tpr_train, linestyle='--', label='Train Set')
  pyplot.plot(adaboost_fpr_validation, adaboost_tpr_validation, linestyle='--', label='Validation Set')
  pyplot.plot(adaboost_fpr_test, adaboost_tpr_test, linestyle='--', label='Test Set')

  # axis labels
  pyplot.xlabel('False Positive Rate')
  pyplot.ylabel('True Positive Rate')

  # show the legend
  pyplot.legend()

  # show the plot
  pyplot.show()