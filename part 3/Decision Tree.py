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


def change_DB_buckets(col, number_of_splits, bucketSize, startpoint):
    for i in range(len(clients_data)):
        for j in range(number_of_splits):
            if (int(clients_data[col].values[i]) <= ((j + 1) * bucketSize) + startpoint):
                # print(clients_data[col].values[i], ((j + 1) * bucketSize) + startpoint , j)
                clients_data[col].values[i] = j
                break


###-------------------Create the data frame-------------------------------###

# remove AI part 3 before submit - i remove !

clients_data = pd.read_csv("DefaultOfCreditCardClients.csv")
clients_data = clients_data.drop(0)  # מוריד שורה ראשונה

# print(clients_data)

###-------------------- Edit paramters in data-set --------------------------###

# LIMIT_BAL
change_DB_buckets('X1', 4, 247500, 10000)

# age
change_DB_buckets('X5', 4, 15, 21)

# BILL_AMT 1-6
change_DB_buckets('X12', 4, 282523, -165580)
change_DB_buckets('X13', 4, 263427, -69777)
change_DB_buckets('X14', 4, 455339, -157264)
change_DB_buckets('X15', 4, 265397, -170000)
change_DB_buckets('X16', 4, 252127, -81334)
change_DB_buckets('X17', 4, 325317, -339603)

# PAY_AMT 1-6
change_DB_buckets('X18', 4, 218388, 0)
change_DB_buckets('X19', 4, 421065, 0)
change_DB_buckets('X20', 4, 224010, 0)
change_DB_buckets('X21', 4, 155250, 0)
change_DB_buckets('X22', 4, 106633, 0)
change_DB_buckets('X23', 4, 132167, 0)

# data of number of options in each Xi

# first 0 is for X0 = null
number_of_X_op = [0, 4, 3, 4, 4, 4, 11, 11, 11, 11, 11, 11, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


# print(clients_data)


###---------------------------- Validation set & Train set & Tree -----לערוך כותרות------------------------###

# יש בעיה בתוכן יש 2 שורות כותרות יעשה בעיות בהמשך
class Node:
    def __init__(self, name, dataM, dataE, sampels, before):
        self.name = name
        self.dataM = dataM
        self.dataE = dataE
        self.sampels = sampels
        self.before = before

# Get all possible options of specific X
def get_option(col_num):
    if (col_num == 2):
        number_of_op = [1, 2]
        return number_of_op
    if (col_num == 6 or col_num == 7 or col_num == 8 or col_num == 9 or col_num == 10 or col_num == 11):
        number_of_op = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        return number_of_op
    else:
        number_of_op = [0, 1, 2, 3]
        return number_of_op

# Get entropy of specific X
def get_one_entropy(col, col_num, data):
    # 1 אנטרופיה מלאה - לא יודע שום דבר על המטבע כלומר לא אומר לי כלום לא משפר אותי
    # ככל שאי הודאות יורדת זה יורד כלומר קטן מ-1 ..
    # אנטרופיה הכי טובה כלומר אני תמיד ידע מה קורה זה אנטרופיה שווה ל-0
    # לוג בבסיס 2
    number_of_op = get_option(col_num)
    #print(np.matrix(number_of_op))
    number_of_0 = [0 for i in range(len(number_of_op))]
    number_of_1 = [0 for i in range(len(number_of_op))]
    number_of_0_p = [0 for i in range(len(number_of_op))]
    number_of_1_p = [0 for i in range(len(number_of_op))]
    entropy_options = [0 for i in range(len(number_of_op))]
    number_of_01 = [0 for i in range(len(number_of_op))]

    for i in range(len(data)):
        k = 0
        for j in number_of_op:
            if (int(data[col].values[i]) == j):
                if (int(data['Y'].values[i]) == 1):
                    number_of_1[k] = number_of_1[k] + 1
                else:
                    number_of_0[k] = number_of_0[k] + 1
                break
            k = k + 1
    #print(np.matrix(number_of_0))
    #print(np.matrix(number_of_1))

    for i in range(len(number_of_0)):
        number_of_01[i] = number_of_0[i] + number_of_1[i]
        if(number_of_01[i]!=0):
            number_of_0_p[i] = number_of_0[i] / number_of_01[i]
            number_of_1_p[i] = number_of_1[i] / number_of_01[i]
        else:
            number_of_0_p[i] = 0
            number_of_1_p[i] = 0
        if (number_of_0[i] == 0 or number_of_1[i] == 0):
            entropy_options[i] = 0
        else:
            entropy_options[i] = round(
                -(number_of_0_p[i] * np.log2(number_of_0_p[i]) + number_of_1_p[i] * np.log2(number_of_1_p[i])), 3)
    #print("in percent")
    #print(np.matrix(number_of_0_p))
    #print(np.matrix(number_of_1_p))
    #print( "Entropy for each =")
    #print(np.matrix(entropy_options))
    #print(" Total values :")
    #print(np.matrix(number_of_01))
    entropy_total = 0
    for i in range(len(number_of_op)):
        entropy_total = entropy_total + entropy_options[i] * (number_of_01[i] / len(data))
    entropy_total = round(entropy_total, 3)
    # print("the X", col_num, "Entropy is :", entropy_total)
    return entropy_total

# get the entropy of all possible splits
def get_entropy_array(data):
    entropy_array = [
        ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20' ,'X21','X22','X23',],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    for i in range (23):
        entropy_array[1][i] = get_one_entropy(entropy_array[0][i] , i+1 , data)
    print(np.matrix(entropy_array))

    min =  entropy_array[1][0]
    x_min = 'X1'
    for i in range (23):
        if(entropy_array[1][i] < min):
            min = entropy_array[1][i]
            x_min = entropy_array[0][i]
    print("The lowest X is :", x_min+',', "his entropy :", min)

def build_tree(k):
    # create x_train , y_train
    # print("Validation set & Train set :")
    x_train = clients_data.drop(['Y'], axis=1).values
    # print(x_train)
    y_train = clients_data['Y'].values
    # print(y_train)

    # split
    x2_train, x_val, y2_train, y_val = train_test_split(x_train, y_train, test_size=k, random_state=123)

    # print("val len = ", len(y_val))
    # print("train len = ", len(y_train))
    get_entropy_array(clients_data)

    # createTreeModelSK(x_train, y_train)


###---------------------------- Kfold -----------------------------###

# need to be function with only K
def tree_error(k, x_train, y_train, x_val, y_val):
    kfold = KFold(n_splits=k, shuffle=True, random_state=123)

    DT_res = pd.DataFrame()
    for train_idx, val_idx in kfold.split(x_train):
        modelDT = DecisionTreeClassifier(criterion='entropy', random_state=123)
        modelDT.fit(x_train[train_idx], y_train[train_idx])
        accTrain = accuracy_score(y_true=y_train[train_idx], y_pred=modelDT.predict(x_train[train_idx]))
        accVal = accuracy_score(y_train[val_idx], modelDT.predict(x_train[val_idx]))
        DT_res = DT_res.append({'accVal': accVal, 'accTrain': accTrain}, ignore_index=True)

    print("Max Depth Tree Performances:")
    print(round(DT_res, 3))
    print(round(DT_res.mean(), 3))

    preds_DT = modelDT.predict(x_val)
    print("Max Depth Tree- Validation accuracy: ", round(accuracy_score(y_val, preds_DT), 3))
    print()


# check tree from sk model
def createTreeModelSK(x_train, y_train):
    # create tree from model
    modelDT = DecisionTreeClassifier(criterion='entropy', random_state=123)
    modelDT.fit(x_train, y_train)
    accTrain = accuracy_score(y_true=y_train, y_pred=modelDT.predict(x_train))
    accVal = accuracy_score(y_train, modelDT.predict(x_train))
    print("Max Depth Tree Performances:")
    print("accVal: ", accVal, ", accTrain: ", accTrain)

    plt.figure(figsize=(30, 10))
    plot_tree(modelDT, filled=True, max_depth=2, class_names=['y=0', 'y=1'],
              feature_names=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                             'PAY_0', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                             'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                             'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
    plt.show()


# main

build_tree(0.2)
# tree_error()
