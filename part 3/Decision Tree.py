import copy
from random import random

from scipy.stats import chisquare, chi2
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
from sympy.stats.drv_types import scipy
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

np.random.seed(356)


class leaf:
    def __init__(self, name, value, data, Entropy, samples, before, options):
        self.name = name
        self.value = value
        self.data = data
        self.Entropy = Entropy
        self.samples = samples
        self.before = before
        self.options = options
        self.next = ""


class parx:
    def __init__(self, name, name_num, Entropydata, Entropy):
        self.name = name
        self.Entropydata = Entropydata
        self.Entropy = Entropy
        self.name_num = name_num


class branch:
    def __init__(self, name, value, leafs):
        self.name = name
        self.value = value
        self.leafs = leafs


def change_DB_buckets(col, number_of_splits, bucketSize, startpoint):
    for i in range(len(clients_data)):
        for j in range(number_of_splits):
            if (int(clients_data[col].values[i]) <= ((j + 1) * bucketSize) + startpoint):
                # print(clients_data[col].values[i], ((j + 1) * bucketSize) + startpoint , j)
                clients_data[col].values[i] = j
                break

def change_DB_buckets2(col, number_of_splits, bucketSize, startpoint , list_to_change):
    for j in range(number_of_splits):
        if (int(list_to_change[col-1]) <= ((j + 1) * bucketSize) + startpoint):
            # print(clients_data[col].values[i], ((j + 1) * bucketSize) + startpoint , j)
            list_to_change[col-1] = str(j)
            break
###-------------------Create the data frame-------------------------------###

# remove AI part 3 before submit - i remove !


clients_data = pd.read_csv("DefaultOfCreditCardClients.csv")
clients_data = clients_data.drop(0)  # מוריד שורה ראשונה
decision_Tree = list()
last_leafs = list()
branches = list()
root = list()
data_list_test = list()
# print(clients_data)

###-------------------- Eding parameters in data-base change into buckets --------------------------###
print("Editing parameters in data-base, change into buckets")
# LIMIT_BAL
change_DB_buckets('X1', 4, 247500, 10000)

# age
change_DB_buckets('X5', 4, 15, 21)

# PAY_0-6
change_DB_buckets('X6', 4, 2, -2)
change_DB_buckets('X7', 4, 2, -2)
change_DB_buckets('X8', 4, 2, -2)
change_DB_buckets('X9', 4, 2, -2)
change_DB_buckets('X10', 4, 2, -2)
change_DB_buckets('X11', 4, 2, -2)

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
##number_of_X_op = [["x1"]0, 4, 2, 4, 4, 4, 11, 11, 11, 11, 10, 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

###---------------------------- Validation set & Train set & Tree -----לערוך כותרות------------------------###

# יש בעיה בתוכן יש 2 שורות כותרות יעשה בעיות בהמשך

# get branch name from leaf
def get_full_branch_name(leaf):
    fname = leaf.before[4:9]
    # print(fname)
    # print(fname[4:5])
    if (fname[4:5] == " "):
        fname = fname[0:4]
    return fname


def split_all_data(k, clients_data2, clients_data3):
    length = 30000 * (1 - k)
    # print(length)
    for i in range(int(length)):
        clients_data2.append(i)
    for i in range(int(length),30000):
        clients_data3.append(i)
    # print(clients_data2)
    # print(len(clients_data2))


# print leaf in list
def print_list(list1):
    st = ""
    for i in list1:
        st = st + i.name + "=" + str(i.value) + " , "
    print(st)


def print_tree():
    for j in decision_Tree:
        str1 = j.name + "=" + str(j.value)
        # print(str1)
        before = j.before
        flag = True
        while (flag):
            for i in decision_Tree:
                name = i.name + "=" + str(i.value)
                if (before == "start"):
                    flag = False
                    break
                # print(name , before)
                if (name == before):
                    str1 = str1 + " -> " + i.name + "=" + str(i.value)
                    # str1 = str1 + " " + i.name + "=" + i.value
                    # print(str1)
                    # str1 = i.name + "=" + i.value
                    before = i.before
        print(str1)


'''
def print_leaf (leaf):
    #print(leaf.name,"=",leaf.value,"e=", leaf.Entropy,"sa=", leaf.samples,"b=", leaf.before)
    #print('"'+leaf.name+'"',',','"'+str(leaf.value)+'"',',','"'+leaf.before+'"')

    #last:
    #print(leaf.name,"=",leaf.value,"b=", leaf.before)

    print(leaf.before[5:]+" -> "+leaf.name)

'''


def sort_by_len(path):
    return len(path.before)


def print_leaf(leaf):
    # print(leaf.name, leaf.value, leaf.Entropy, leaf.samples, leaf.before)
    print(leaf.before[4:] + " -> " + leaf.name + "=" + str(leaf.value))


# Get all possible options of specific X
def get_option(col_num):
    if (col_num == 2):
        number_of_op = [1, 2]
        return number_of_op
    else:
        number_of_op = [0, 1, 2, 3]
        return number_of_op


# Get entropy of specific X (מייצרת ParX)
def get_one_entropy(col, col_num, data):
    # 1 אנטרופיה מלאה - לא יודע שום דבר על המטבע כלומר לא אומר לי כלום לא משפר אותי
    # ככל שאי הודאות יורדת זה יורד כלומר קטן מ-1 ..
    # אנטרופיה הכי טובה כלומר אני תמיד ידע מה קורה זה אנטרופיה שווה ל-0
    # לוג בבסיס 2
    number_of_op = get_option(col_num)
    # print(np.matrix(number_of_op))
    number_of_0 = [0 for i in range(len(number_of_op))]
    number_of_1 = [0 for i in range(len(number_of_op))]
    number_of_0_p = [0 for i in range(len(number_of_op))]
    number_of_1_p = [0 for i in range(len(number_of_op))]
    entropy_options = [0 for i in range(len(number_of_op))]
    number_of_01 = [0 for i in range(len(number_of_op))]

    for i in data:
        k = 0
        for j in number_of_op:
            if (int(clients_data[col].values[i]) == j):
                if (int(clients_data['Y'].values[i]) == 1):
                    number_of_1[k] = number_of_1[k] + 1
                else:
                    number_of_0[k] = number_of_0[k] + 1
                break
            k = k + 1
    # print(np.matrix(number_of_0))
    # print(np.matrix(number_of_1))

    for i in range(len(number_of_0)):
        number_of_01[i] = number_of_0[i] + number_of_1[i]
        if (number_of_01[i] != 0):
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

    '''print("in percent")
    print(np.matrix(number_of_0_p))
    print(np.matrix(number_of_1_p))
    print( "Entropy for each =")
    print(np.matrix(entropy_options))
    print(" Total values :")
    print(np.matrix(number_of_01))'''
    entropy_total = 0
    for i in range(len(number_of_op)):
        entropy_total = entropy_total + entropy_options[i] * (number_of_01[i] / len(data))
    entropy_total = round(entropy_total, 3)
    # print("the X", col_num, "Entropy is :", entropy_total)
    return parx(col, col_num, entropy_options, entropy_total)


# Generate the entropy of all possible splits and return the lowest
def get_entropy_array(options, data_rows):
    # אין לי שימוש במערך הזה, וב entropy_array[1][i-1] = nodex.Entropy
    # אם לא צריך להדפיס את כולם או שיט כזה למחוק.
    entropy_array = [
        ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17',
         'X18', 'X19', 'X20', 'X21', 'X22', 'X23', ],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
    min = 2
    x_min = 'A'
    x_min_num = -2
    for i in options:
        nodex = get_one_entropy(entropy_array[0][i - 1], i, data_rows)  # parx
        entropy_array[1][i - 1] = nodex.Entropy
        if (nodex.Entropy < min):
            nodex2 = copy.deepcopy(nodex)
            min = nodex.Entropy
            x_min = nodex.name
            x_min_num = i
    # print(np.matrix(entropy_array))
    '''if(x_min=="X12"):
        print(nodex2.name , nodex2.Entropy , nodex2.Entropydata)'''
    # print("The lowest X is :", x_min+',', "his entropy :", min)
    if (min == 2):
        return 2
    options.remove(x_min_num)
    # print(options)
    return nodex2


# אם אני רוצה להגיע לשורה הנכונה פשוט data_rows=data_new
# print(clients_data.values[data_new[0]])

# split data for leaf in the tree
def split_data(colX, index, row_data):
    data_rows = list()
    for i in row_data:
        if (i == 30000):
            break
        if (int(clients_data[colX].values[i]) == index):
            # print( int(clients_data['Unnamed: 0'].values[i]), int(clients_data['X3'].values[i]), int(clients_data[colX].values[i]))
            data_rows.append(i)

    # print(len(data_rows))
    return data_rows


def make_leafs_first(row_data, colX, col_num, Entropy_data, before_name, options):
    # clientsdata , 'X6' , 6, [0.4,0.3....]
    number_of_op = get_option(col_num)
    k = 0
    for i in number_of_op:
        data_leaf = split_data(colX, i, row_data)
        if (len(data_leaf) > 0):
            optionsleaf = copy.deepcopy(options)
            decision_Tree.append(leaf(colX, i, data_leaf, Entropy_data[k], len(data_leaf), before_name, optionsleaf))
        k = k + 1


# Create leafs in the tree fFrom specific Xi
def make_leafs(row_data, colX, col_num, Entropy_data, before_name, current_leaf, pathleaf, options):
    # clientsdata , 'X6' , 6, [0.4,0.3....]
    number_of_op = get_option(col_num)
    k = 0
    t = 0
    # print("make leaf  ,", colX,"b=" ,before_name)
    for i in number_of_op:
        data_leaf = split_data(colX, i, row_data)
        if (len(data_leaf) > 0):
            optionsleaf = copy.deepcopy(options)
            decision_Tree.append(leaf(colX, i, data_leaf, Entropy_data[k], len(data_leaf), before_name, optionsleaf))
            # print(colX,"=",i)
            if (t != 0):  # אם זה לא העלה הראשון תוסיף למאגר שצריך לבדוק
                pathleaf.append(leaf(colX, i, data_leaf, Entropy_data[k], len(data_leaf), before_name, optionsleaf))
            if (t == 0):  # כדאי שאוכל להחזיר את העלה הראשון
                current_leaf = leaf(colX, i, data_leaf, Entropy_data[k], len(data_leaf), before_name, optionsleaf)
            t = t + 1
        k = k + 1
    return current_leaf


def change_next(leaf_other):
    for leaf in decision_Tree:
        if (leaf.before == leaf_other.before and leaf.name == leaf_other.name and leaf.value == leaf_other.value):
            leaf.next = "last leaf"


def find_last_leafs(branch_name):
    for leaf in decision_Tree:
        bname = get_full_branch_name(leaf)
        # print(bname , branch_name ,bname == branch_name)
        if (leaf.next == "last leaf" and bname == branch_name):
            leafi = copy.deepcopy(leaf)
            last_leafs.append(leafi)
            # print_leaf(leafi)

    # print("last_leafs len :",len(last_leafs))
    '''for leaf in last_leafs:
        print(leaf.name , leaf.value , leaf.next)'''


def leaf_allready_inside(list1, leaf_to_check):
    for leaf in list1:
        if (
                leaf.before == leaf_to_check.before and leaf.name == leaf_to_check.name and leaf.value == leaf_to_check.value):
            return True
    return False


def get_longest():
    max = len(last_leafs[0].before)
    longest_leaf = 0
    for i in range(len(last_leafs)):
        if (max < len(last_leafs[i].before)):
            max = len(last_leafs[i].before)
            longest_leaf = i
    return longest_leaf


def get_number_of_splits(xname):
    xnum = xname.split("X")
    return len(get_option(int(xnum[1])))


def cut_leaf(leaf_cut):
    before = ""
    for leaf in decision_Tree:
        if (leaf_cut.before == leaf.before):
            # print(leaf.before)
            before = leaf.before[:-9]
            decision_Tree.remove(leaf)
            # print("remove decision_Tree")
            # print(leaf in decision_Tree)
    # print("123 before", before)
    for leaf in decision_Tree:
        # print(leaf.before)
        if (leaf.before == before):
            leaf.next = "last leaf"
            if (leaf_allready_inside(last_leafs, leaf) == False):
                last_leafs.append(copy.deepcopy(leaf))
                # print("add")
    for leaf in last_leafs:
        if (leaf_cut.before == leaf.before):
            last_leafs.remove(leaf)
            # print("remove last_leafs")
            # print(leaf in last_leafs)


def check_meaning_leaf():
    # print(leaf_i.name , leaf_i.data)
    couter = 0
    for i in range(len(last_leafs)):
        couter += 1
        index = get_longest()
        longest_leaf = last_leafs[index]
        # print(" longest name :", longest_leaf.name,longest_leaf.value, "before = ", longest_leaf.before)
        # lastNode = longest_leaf.before.split(" -> ")
        # print(lastNode[-1])
        need_to_cut = chi_test(longest_leaf)
        if (need_to_cut):
            # print("cut")
            cut_leaf(longest_leaf)
        else:
            # print("find one on , finish (need to check")
            # print("number of small branches = ", len(last_leafs))
            # צריך לסדר כי לא בטוח יהיה X6 ולא בטוח ערך 0 !
            # last_leafs.sort(key=sort_by_len)
            branchfullname = get_full_branch_name(last_leafs[-1])
            branches.append(branch(branchfullname[0:2], branchfullname[3:], copy.deepcopy(last_leafs)))
            # print("name = ", branches[0].name, "value =", branches[0].value)
            break
    print("The splits in the branch are:")
    if (len(last_leafs) == 0):
        print("No more splits in this branch")
    for leaf in last_leafs:
        print_leaf(leaf)
    last_leafs.clear()

# check pruning chi^2 test
def chi_test(longest_leaf):
    list_last_paths = list()
    for leaf in decision_Tree:
        if (longest_leaf.before in leaf.before):
            list_last_paths.append(copy.deepcopy(leaf))
            # print(leaf.name, "=", leaf.value, " | s= ", leaf.samples, " | b=", leaf.before, )
    pn = [[0 for j in range(4)] for i in range(len(list_last_paths))]
    delta = [0 for i in range(len(list_last_paths))]
    totalcounter0 = 0
    totalcounter1 = 0
    for i in range(len(list_last_paths)):
        pn[i][0] = 0
        pn[i][1] = 0
        for row in list_last_paths[i].data:
            if (clients_data['Y'].values[row] == "0"):
                pn[i][0] += 1  # counter0 = pk
            if (clients_data['Y'].values[row] == "1"):
                pn[i][1] += 1  # counter1 =nk
        totalcounter0 += pn[i][0]  # totalcounter0 = p
        totalcounter1 += pn[i][1]  # totalcounter1 = n
    # print(totalcounter0, totalcounter1)
    # print(np.matrix(pn))

    statisti = 0
    for i in range(len(list_last_paths)):
        pn[i][2] = round((totalcounter0 * (pn[i][0] + pn[i][1])) / (totalcounter0 + totalcounter1),
                         3)  # p*(pk+pn)/(p+n) = p^
        pn[i][3] = round((totalcounter1 * (pn[i][0] + pn[i][1])) / (totalcounter0 + totalcounter1),
                         3)  # n*(pk+pn)/(p+n) = n^
        if (pn[i][3] == 0):
            delta[i] = round(delta[i] + (((pow((pn[i][0] - pn[i][2]), 2)) / pn[i][2])), 5)  # delta
        if (pn[i][2] == 0):
            delta[i] = round(delta[i] + (((pow((pn[i][1] - pn[i][3]), 2)) / pn[i][3])), 5)  # delta
        if (pn[i][2] != 0 and pn[i][3] != 0):
            delta[i] = delta[i] + (((pow((pn[i][0] - pn[i][2]), 2)) / pn[i][2]) + (
                        (pow((pn[i][1] - pn[i][3]), 2)) / pn[i][3]))  # delta
        statisti += delta[i]
    # print(np.matrix(pn))
    # print(pn[0][0])
    # print(np.matrix(delta))

    dgree_of_free = get_number_of_splits(longest_leaf.name)
    criti = chi2.ppf(0.95, dgree_of_free)
    # print("statisti= ", statisti ,"criti = ",criti , "s<c = remove")
    if (statisti < criti):
        return True
    else:
        return False

#create list in order to use will_default
def get_list_from_client(row):
    temp1 = list()
    #st = ""
    for i in range(23):
        #st += str((clients_data['X'+str(i+1)].values[row])) + ","
        temp1.append(str(clients_data['X'+str(i+1)].values[row]))
    #print(st)
    return temp1

# Build all the tree
def build_tree(k):
    print("Start Building the Tree")
    # create x_train , y_train
    # print("Validation set & Train set :")
    # x_train = clients_data.drop(['Y'], axis=1).values
    # print(x_train)
    # y_train = clients_data['Y'].values
    # print(y_train)

    # split
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=k, random_state=123)
    # print("val len = ", len(y_val))
    # print("train len = ", len(y_train))
    clients_data2 = list()
    split_all_data(k,data_list_test, clients_data2)
    options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    #data_rows = list()
    #for i in clients_data2:
        #data_rows.append(i)
    nodex = get_entropy_array(options, clients_data2)
    print("The root of the tree is :", nodex.name)
    root.append(nodex.name)
    make_leafs_first(clients_data2, nodex.name, nodex.name_num, nodex.Entropydata, "start", options)
    #print("df len = ", len(decision_Tree))
    pathleaf = list()
    k = 0
    # print(len(get_option(decision_Tree[1].name[1:2])))
    t = 0
    for i in range(len(decision_Tree)):
        print("The branch is :", decision_Tree[i].name, "=", decision_Tree[i].value)
        options_leaf = copy.deepcopy(options)
        pathleaf.clear()
        finish_path = False
        # pathleaf.append(i)
        current_leaf = decision_Tree[i]
        before_name = ""

        while (finish_path == False):
            if (current_leaf.Entropy == 0):
                change_next(current_leaf)
                # print(current_leaf.name,current_leaf.next)
                # print("entropy = 0 ", current_leaf.name,"=", current_leaf.value,"his e= ",current_leaf.Entropy)
                if (len(pathleaf) == 0):
                    break
                # current_leaf = make_leafs(current_leaf.data, nodex.name, nodex.name_num, nodex.Entropydata, before_name , current_leaf , pathleaf,options_leaf)
                current_leaf = pathleaf[-1]
                pathleaf.remove(pathleaf[-1])
                if (len(current_leaf.options) == 0):
                    change_next(current_leaf)
                options_leaf = copy.deepcopy(current_leaf.options)
                before_name = current_leaf.before

            else:
                nodex = get_entropy_array(options_leaf, current_leaf.data)
                if (nodex == 2):
                    while (True):
                        if (len(pathleaf) == 0):
                            break
                        current_leaf = pathleaf[-1]
                        pathleaf.remove(pathleaf[-1])
                        if (len(current_leaf.options) == 0):
                            change_next(current_leaf)
                        options_leaf = copy.deepcopy(current_leaf.options)
                        nodex = get_entropy_array(options_leaf, current_leaf.data)
                        before_name = current_leaf.before
                        # print("inside while")
                        # print(current_leaf.name ," = ",current_leaf.value ,current_leaf.options)
                        # print(nodex)
                        if (nodex != 2):
                            break
                # print(current_leaf.name, " = ", current_leaf.value, current_leaf.options)
                # print(nodex)
                if (nodex == 2):
                    break
                before_name = before_name + " -> " + current_leaf.name + "=" + str(current_leaf.value)
                current_leaf = make_leafs(current_leaf.data, nodex.name, nodex.name_num, nodex.Entropydata, before_name,
                                          current_leaf, pathleaf, options_leaf)

                # print_list(pathleaf)
                # print(current_leaf.name , "=" , current_leaf.value ,", ", current_leaf.before)

        find_last_leafs(decision_Tree[i].name + "=" + str(decision_Tree[i].value))
        check_meaning_leaf()

    # print_tree()

    # name, value, data, Entropy, samples, before

    # createTreeModelSK(x_train, y_train)
    #print(data_list_test)
    print("Calculate the quality of the decision tree with 8 k folds :")
    tree_error(8)
    #for row in data_list_test:
    #    def_to_check = get_list_from_client(row)
    #    will_default(def_to_check)




###---------------------------- will_default-----------------------------###

def change_parm(list):
    change_DB_buckets2(1, 4, 247500, 10000, list)
    change_DB_buckets2(5, 4, 15, 21, list)
    change_DB_buckets2(6, 4, 2, -2, list)
    change_DB_buckets2(7, 4, 2, -2, list)
    change_DB_buckets2(8, 4, 2, -2, list)
    change_DB_buckets2(9, 4, 2, -2, list)
    change_DB_buckets2(10, 4, 2, -2, list)
    change_DB_buckets2(11, 4, 2, -2, list)
    change_DB_buckets2(12, 4, 282523, -165580, list)
    change_DB_buckets2(13, 4, 263427, -69777, list)
    change_DB_buckets2(14, 4, 455339, -157264, list)
    change_DB_buckets2(15, 4, 265397, -170000, list)
    change_DB_buckets2(16, 4, 252127, -81334, list)
    change_DB_buckets2(17, 4, 325317, -339603, list)
    change_DB_buckets2(18, 4, 218388, 0,list)
    change_DB_buckets2(19, 4, 421065, 0, list)
    change_DB_buckets2(20, 4, 224010, 0, list)
    change_DB_buckets2(21, 4, 155250, 0, list)
    change_DB_buckets2(22, 4, 106633, 0, list)
    change_DB_buckets2(23, 4, 132167, 0, list)
    #print(list)

def convert_list_int_to_str(list1):
    string_ints = [str(int) for int in list1]
    str_of_ints = ",".join(string_ints)
    splits = str_of_ints.split(",")
    list_temp = list()
    for i in splits:
        list_temp.append(i)
    #print(list_temp)
    return list_temp

def str_or_int(list1):
    if ((str(type(list1[0])) == "<class 'str'>")):
        return True
    if ((str(type(list1[0])) == "<class 'int'>")):
        return False

# get the most common y option of this branch
def get_Y(leaf, bool):
    if(bool): # its leaf
        number0 = 0
        number1 = 0
        for row in leaf.data:
            if (clients_data['Y'].values[row] == "0"):
                number0 += 1
            if (clients_data['Y'].values[row] == "1"):
                number1 += 1

        #print(number0,number1, number0+number1)
        if(number0 == 0):
            return 1
        if (number1 == 0):
            return 0
        rand_num = random()
        #print(rand_num)
        if(rand_num <= (number0/(number0+number1)) ):
            return 0
        else:
            return 1
    else: # list
        #print_list(leaf)
        number0 = 0
        number1 = 0
        for leafi in leaf:
            for row in leafi.data:
                if (clients_data['Y'].values[row] == "0"):
                    number0 += 1
                if (clients_data['Y'].values[row] == "1"):
                    number1 += 1
        #print(number0, number1, number0 + number1)
        if (number0 == 0):
            return 1
        if (number1 == 0):
            return 0
        rand_num = random()
        #print(rand_num)
        if (rand_num <= (number0 / (number0 + number1))):
            return 0
        else:
            return 1

# get the leaf in the last
def get_branch_value(before , before2 ,branch , name ):
    for leaf in branch.leafs:
        #print(leaf.before,"|",before,"|",leaf.before == before ,"|",leaf.name+"="+str(leaf.value),"|", name,"|", leaf.name+"="+str(leaf.value) == name)
        if (leaf.before == before and leaf.name+"="+str(leaf.value) == name):
            return get_Y(copy.deepcopy(leaf), True)
    list_x = list()
    for leaf in branch.leafs:
        if ( before2 in leaf.before):
            list_x.append(copy.deepcopy(leaf))
    return get_Y(list_x, False)

# get next split in the tree
def get_value_from_branch(before , branch):
    for leaf in branch.leafs:
        if(leaf.before == before):
            return leaf.name
    return "none"

def will_default(list):
    if (str_or_int(list)):
        #print("Enter will_default")
        change_parm(list)
        rootname = root[-1]
        rootXnum = rootname[-1]
        #print("rootXnum=", rootXnum)
        X_branch_value = list[int(rootXnum)-1]
        #print("branch_value = " , X_branch_value)
        for branch in branches:
            if(branch.value == X_branch_value):
                #print("find the branch", branch.name , "=" , branch.value)
                before = " -> "+branch.name+"="+str(branch.value)
                #print("before start =", before)
                for i in range(23):
                    next_X = get_value_from_branch(before, branch)
                    if(next_X != "none"):
                        before += " -> " + next_X + "=" + list[int(next_X[1:])-1]
                        #print("before B =", before)
                    else:
                        #print("last before = ", before)
                        X_arr = before.split(" -> ")
                        before = ""
                        for i in X_arr[1:len(X_arr)-1]:
                            before += " -> "+i
                        #print("last before cut last = ", before)
                        #Yval = get_branch_value(before,before+" -> "+X_arr[len(X_arr)-2], branch , X_arr[len(X_arr)-2])
                        Yval = get_branch_value(before,before, branch , X_arr[len(X_arr)-1])
                        #print(leaf.name,"=" ,leaf.value)
                        return Yval

        # no branch
        rand_num = random()
        if(rand_num>0.5):
            return 0
        return 1
    else : # list of ints
        list_temp = convert_list_int_to_str(list)
        return will_default(list_temp)



###---------------------------- Error - Kfold -----------------------------###

# K fold splitting
def KFoldList(data_list, k, k_data_list):
  lengthk = int(len(data_list) / k)
  last = len(data_list) - lengthk * (k - 1)
  temp_list = list()
  t = 1
  for listi in range (k-1):
    for i in range(((t - 1) * lengthk), (t * lengthk)):
      temp_list.append(i)
    k_data_list.append(copy.deepcopy(temp_list))
    temp_list.clear()
    t=t+1
  for i in range(((t - 1) * lengthk), ((t - 1) * lengthk) + last ):
    temp_list.append(i)
  k_data_list.append(copy.deepcopy(temp_list))



# need to be function with only K
def tree_error(k):
    print("Start calculate the quality of the decision tree:")
    #kfold = KFold(n_splits=k, shuffle=True, random_state=123)
    k_data_list = list()
    KFoldList(data_list_test ,k ,k_data_list)
    #print("sets")
    #print(k_data_list)
    Total_results = list()
    k_num = 1
    for listi in k_data_list:
        errors = 0
        good_pred = 0
        for row in listi:
            real_val = clients_data['Y'].values[row]
            def_to_check = get_list_from_client(row)
            pred_val = str(will_default(def_to_check))
            #print(pred_val,real_val,pred_val == real_val)
            if(pred_val == real_val):
                good_pred += 1
            else:
                errors += 1
        result = list()
        result.append(good_pred)
        result.append(errors)
        result.append(round( (good_pred/(good_pred+errors)), 3))
        #print("finish")
        #print(result)
        Total_results.append(copy.deepcopy(result))
        result.clear()
        print("The fold number" , k_num , "the acc is:" , round( (good_pred/(good_pred+errors)), 3))
        k_num += 1
    total_acc = 0;
    for foldk_list in Total_results:
        total_acc+=foldk_list[2]
    #print((Total_results))
    print("The Tree acc is :", round((total_acc/k),3))

'''
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

'''
# main

build_tree(0.6)
'''
list12 = ["20000", "2", "2", "1", "24", "1", "2", "-1", "3", "0", "-2", "3913", "3102", "689", "0", "0", "0",
                "0", "689", "0", "0", "0", "0"]  # y ="1" , X6=1
list123 = [20000,2,2,1,24,1,2,-1,3,0,-2,3913,3102,689,0,0,0,0,689,0,0,0,0,1]
list1234 =[20000,2,2,1,24,2,2,-1,-1,-2,-2,3913,3102,689,0,0,0,0,689,0,0,0,0,1]
ans = will_default(list12)
print("y = " , ans)
ans2 = will_default(list123)
print("y = " , ans2)
ans2 = will_default(list1234)
print("y = " , ans2)
# tree_error()
'''
