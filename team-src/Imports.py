#Import the other files 

import pandas as pd
import numpy as np
import collections
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import PCA

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

#for the NN
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
import tensorflow as tf
from keras.optimizers import SGD, RMSprop
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score

#for inbalance issues 
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from collections import Counter
from matplotlib import pyplot 
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

#imports for XGBoost
import xgboost as xgb 
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_tree
from yellowbrick.classifier import ClassBalance, ROCAUC, ClassificationReport, ClassPredictionError
from keras.utils.vis_utils import plot_model


from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE