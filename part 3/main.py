import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


np.random.seed(356)

###-------------------Create the data frame-------------------------------###

# remove AI part 3 before submit !

clients_data=pd.read_csv("AI part 3\DefaultOfCreditCardClients.csv")
clients_data=clients_data.drop(0)
#print(clients_data)

###---------------------------- Validation set & Train set & Tree -----לערוך כותרות------------------------###

# יש בעיה בתוכן יש 2 שורות כותרות יעשה בעיות בהמשך

def build_tree(k):

    # create x_train , y_train
    print("Validation set & Train set :")
    x_train = clients_data.drop(['Y'], axis=1).values
    #print(x_train)
    y_train = clients_data['Y'].values
    #print(y_train)


    # split
    x2_train, x_val, y2_train, y_val = train_test_split(x_train, y_train, test_size=k, random_state=123)

    print("val len = " , len(y_val))
    print("train len = " , len(y_train))

    modelDT = DecisionTreeClassifier(criterion='entropy', random_state=123)
    modelDT.fit(x_train, y_train)
    accTrain = accuracy_score(y_true=y_train, y_pred=modelDT.predict(x_train))
    accVal = accuracy_score(y_train, modelDT.predict(x_train))
    print("Max Depth Tree Performances:")
    print("accVal: ", accVal, ", accTrain: " ,accTrain)


    plt.figure(figsize=(30, 10))
    plot_tree(modelDT, filled=True, max_depth=10, class_names=['y=0', 'y=1'],
              feature_names=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                             'PAY_0', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                             'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                             'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
    plt.show()




###---------------------------- Kfold -----------------------------###

# need to be function with only K
def tree_error(k,x_train,y_train,x_val, y_val):

    kfold = KFold(n_splits=k, shuffle=True, random_state=123)

    DT_res = pd.DataFrame()
    for train_idx, val_idx in kfold.split(x_train):
           modelDT = DecisionTreeClassifier(criterion='entropy', random_state=123)
           modelDT.fit(x_train[train_idx], y_train[train_idx])
           accTrain=accuracy_score(y_true=y_train[train_idx], y_pred=modelDT.predict(x_train[train_idx]))
           accVal = accuracy_score(y_train[val_idx], modelDT.predict(x_train[val_idx]))
           DT_res = DT_res.append({'accVal': accVal , 'accTrain' : accTrain}, ignore_index=True)

    print("Max Depth Tree Performances:")
    print(round(DT_res,3))
    print(round(DT_res.mean(),3))

    preds_DT = modelDT.predict(x_val)
    print("Max Depth Tree- Validation accuracy: ", round(accuracy_score(y_val, preds_DT), 3))
    print()

build_tree(0.2)