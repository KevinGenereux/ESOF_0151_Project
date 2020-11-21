metrics.py
K
I
L
T
Type
Text
Size
2 KB (2,057 bytes)
Storage used
2 KB (2,057 bytes)
Location
Project Code
Owner
me
Modified
Nov 13, 2020 by me
Opened
Nov 13, 2020 by me
Created
Oct 30, 2020 with Google Drive File Stream
Add a description
Viewers can download
from sklearn.metrics import confusion_matrix, auc, roc_curve
import matplotlib.pyplot as plt

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[3]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred).ravel()[2]

# FP / (FP + TN)
def specificity(y_true, y_pred):
    if tn(y_true, y_pred) + fp(y_true, y_pred) == 0:
        return 0
    else:
        return float(tn(y_true, y_pred)) / float((tn(y_true, y_pred) + fp(y_true, y_pred)))

# TP / (TP + FN)
def sensitivity(y_true, y_pred):
    if tp(y_true, y_pred) + fn(y_true, y_pred) == 0:
        return 0
    else:
        return float(tp(y_true, y_pred)) / float((tp(y_true, y_pred) + fn(y_true, y_pred)))

# TP / (TP + 0.5(FP + FN))
def f1(y_true, y_pred):
    if precision(y_true, y_pred) + sensitivity(y_true, y_pred) == 0:
        return 0
    else:
        return 2.0 * float(precision(y_true, y_pred) * sensitivity(y_true, y_pred)) / (precision(y_true, y_pred) +
                                                                                sensitivity(y_true, y_pred))
# (TP + FN) / (TP + TN + FP + FN)
def accuracy(y_true, y_pred):
    return float(tp(y_true, y_pred) + tn(y_true, y_pred)) / float((tp(y_true, y_pred) + tn(y_true, y_pred) + fp(y_true, y_pred)
                                                         + fn(y_true, y_pred)))
# generate an ROC curve graph
def roc(y_true, y_pred, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkgreen',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('ROCs/' + filename + '.png')

