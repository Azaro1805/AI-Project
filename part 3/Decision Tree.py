import copy
from random import random
from scipy.stats import chisquare, chi2
import numpy as np
import pandas as pd

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


###---------------------------- Initialization-----------------------------###

clients_data = pd.read_csv("DefaultOfCreditCardClients.csv")
clients_data = clients_data.drop(0)
decision_Tree = list()
last_leafs = list()
branches = list()
root = list()
data_list_test = list()
data_list_test2 = list()

###-------------------- Eding parameters in data-base change into buckets --------------------------###
print()
print("Editing parameters in data-base, change into buckets")
print()

###---------------------------- Data into buckets-----------------------------###

# Change data in order to get into buckets - when the data in data frame
def change_DB_buckets(col, number_of_splits, bucketSize, startpoint):
    for i in range(len(clients_data)):
        for j in range(number_of_splits):
            if (int(clients_data[col].values[i]) <= ((j + 1) * bucketSize) + startpoint):
                # print(clients_data[col].values[i], ((j + 1) * bucketSize) + startpoint , j)
                clients_data[col].values[i] = j
                break

# Change data in order to get into buckets after first time
def change_DB_buckets2(col, number_of_splits, bucketSize, startpoint , list_to_change):
    for j in range(number_of_splits):
        if (int(list_to_change[col-1]) <= ((j + 1) * bucketSize) + startpoint):
            # print(clients_data[col].values[i], ((j + 1) * bucketSize) + startpoint , j)
            list_to_change[col-1] = str(j)
            break

# LIMIT_BAL
change_DB_buckets('X1', 4, 247500, 10000)

# Age
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

###---------------------------- Generate Tree -----------------------------###

# Get branch name from leaf
def get_full_branch_name(leaf):
    fname = leaf.before[4:9]
    # print(fname)
    # print(fname[4:5])
    if (fname[4:5] == " "):
        fname = fname[0:4]
    return fname

# split all data to 2 set - train and test
def split_all_data(k, test, clients_data2 , clients_data):
    length = len(clients_data) * (1 - k)
    # print(length)
    for i in range(int(length)):
        test.append(i)
    for i in range(int(length),len(clients_data)):
        clients_data2.append(i)
    # print(clients_data2)
    # print(len(clients_data2))

# Print leaf in list
def print_list(list1):
    st = ""
    for i in list1:
        st = st + i.name + "=" + str(i.value) + " , "
    print(st)

# Print the tree leafs
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

# Sort list the path length
def sort_by_len(path):
    return len(path.before)

# Print leaf and his path
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

# Split data for leaf in the tree
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

# Make the first leaf of the tree
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

# Change next value of leaf
def change_next(leaf_other):
    for leaf in decision_Tree:
        if (leaf.before == leaf_other.before and leaf.name == leaf_other.name and leaf.value == leaf_other.value):
            leaf.next = "last leaf"

# Get the last leaf from the branch
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

# Check if the leaf is inside the list by before and value
def leaf_allready_inside(list1, leaf_to_check):
    if(len(list1)==0):
        return False
    for leaf in list1:
        if (leaf.before == leaf_to_check.before and leaf.name == leaf_to_check.name and leaf.value == leaf_to_check.value):
            return True
    return False

# Get the longest path of leaf from list
def get_longest():
    max = len(last_leafs[0].before)
    longest_leaf = 0
    for i in range(len(last_leafs)):
        if (max < len(last_leafs[i].before)):
            max = len(last_leafs[i].before)
            longest_leaf = i
    return longest_leaf

# Get the number of options of Xi
def get_number_of_splits(xname):
    xnum = xname.split("X")
    return len(get_option(int(xnum[1])))

# Cut leaf from list and tree
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

# Check if leaf is inside list only before
def leaf_allready_inside2(list1, leaf_to_check):
    if(len(list1)==0):
        return False
    #print("start check inside good list")
    for leaf in list1:
        #print(leaf_to_check.before ,"|", leaf.before ,"|", leaf_to_check.before in leaf.before)
        if (leaf_to_check.before in leaf.before):
            return True
    return False

# Check if leaf is inside list by before & name
def leaf_allready_inside3(list1, leaf_to_check):
    if(len(list1)==0):
        return False
    #print("start check inside good list")
    for leaf in list1:
        #print(leaf_to_check.before ,"|", leaf.before ,"|", leaf_to_check.before in leaf.before)
        full_b = leaf_to_check.before+" -> "+leaf_to_check.name+"="+str(leaf_to_check.value)
        #print(full_b)
        if (full_b in leaf.before):
            return True
    return False

# Generate branches
def check_meaning_leaf():
    # print(leaf_i.name , leaf_i.data)
    good_path = list()
    couter = 0
    flag = True

    while(flag):
        counter = 0
        for i in range(len(last_leafs)):
            if(len(last_leafs)==0):
                break
            couter += 1
            index = get_longest()
            longest_leaf = last_leafs[index]
            if(leaf_allready_inside2(good_path, longest_leaf) == False):
                # print(" longest name :", longest_leaf.name,longest_leaf.value, "before = ", longest_leaf.before)
                # lastNode = longest_leaf.before.split(" -> ")
                # print(lastNode[-1])
                need_to_cut = chi_test(longest_leaf)
                if (need_to_cut):
                    #print("cut")
                    cut_leaf(longest_leaf)
                else:
                    good_path.append(copy.deepcopy(longest_leaf))
                    for leaf1 in last_leafs:
                        if (longest_leaf.before == leaf1.before):
                            last_leafs.remove(leaf1)
                    counter += 1
                    break
            else:
                good_path.append(copy.deepcopy(longest_leaf))
                for leaf1 in last_leafs:
                    if (longest_leaf.before == leaf1.before):
                        last_leafs.remove(leaf1)
                counter += 1
                break
        if(counter == 0):
            #print("endddddddddddddddddddd")
            flag = False
            break

                            # print("remove last_leafs")
                            # print(leaf in last_leafs)
                    #print("good_path = ", good_path[-1].before)
                    # print("find one on , finish (need to check")
                    # print("number of small branches = ", len(last_leafs))
                    # צריך לסדר כי לא בטוח יהיה X6 ולא בטוח ערך 0 !
                    # last_leafs.sort(key=sort_by_len)
                    # print("name = ", branches[0].name, "value =", branches[0].value)
    #print("finish")
    for leaf in good_path:
        if (leaf_allready_inside3(last_leafs, leaf) == False):
            last_leafs.append(copy.deepcopy(leaf))
    #print("good path , finish")
    '''for i in good_path:
        print(i.before)
    print(len(last_leafs))'''
    if (len(last_leafs) == 0):
        print("The branch is been deleted")
    else:
        branchfullname = get_full_branch_name(last_leafs[0])
        #print("branchfullname", branchfullname)
        branches.append(branch(branchfullname[0:2], branchfullname[3:], copy.deepcopy(last_leafs)))
        print("The splits in the branch are:")

    for leaf in last_leafs:
        print_leaf(leaf)
    last_leafs.clear()
    good_path.clear()

# Check pruning chi^2 test
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
    #print("statisti= ", statisti ,"criti = ",criti , "s<c = remove")
    if (statisti < criti):
        return True
    else:
        return False

# Create list in order to use will_default
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
    decision_Tree.clear()
    data_list_test.clear()
    last_leafs.clear()
    branches.clear()
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
    split_all_data(k,data_list_test, clients_data2 ,clients_data)
    #print(data_list_test)
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
    #k = 0
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
    if(k != 1):
        print("Calculate the quality of the decision tree:")
        #data_list_test
        errors = 0
        good_pred = 0
        for row in data_list_test:
            #print(row)
            real_val = clients_data['Y'].values[row]
            def_to_check = get_list_from_client(row)
            pred_val = str(will_default2(def_to_check))
            #print(pred_val,real_val,pred_val == real_val)
            if(pred_val == real_val):
                good_pred += 1
            else:
                errors += 1
        #print(good_pred,errors)
        #print(good_pred/(good_pred+errors))
        if (good_pred == 0):
            print("The Tree acc is :", 0)
            print("The Tree error is :", 1)
            return 0
        result = round((good_pred / (good_pred + errors)), 3)
        print("The Tree acc is :", result)
        print("The Tree error is :", round((errors / (good_pred + errors)), 3))
        return round((errors / (good_pred + errors)), 3)
        #for row in data_list_test:
        #    def_to_check = get_list_from_client(row)
        #    will_default(def_to_check)

###---------------------------- will_default-----------------------------###

# Change values to the buckets value
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

# Convert list of int to list of string
def convert_list_int_to_str(list1):
    string_ints = [str(int) for int in list1]
    str_of_ints = ",".join(string_ints)
    splits = str_of_ints.split(",")
    list_temp = list()
    for i in splits:
        list_temp.append(i)
    #print(list_temp)
    return list_temp

# Check if list contain objects of int or string
def str_or_int(list1):
    if ((str(type(list1[0])) == "<class 'str'>")):
        return True
    if ((str(type(list1[0])) == "<class 'int'>")):
        return False

# Get the most common y option of this branch
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

# Get the leaf in the last
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

# Get next split in the tree
def get_value_from_branch(before , branch):
    for leaf in branch.leafs:
        if(leaf.before == before):
            return leaf.name
    return "none"

# Will default on current tree
def will_default2(list):
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
        return will_default2(list_temp)

# will default for full tree
def will_default(list):
    if (str_or_int(list)):
        build_tree(1)
        print("Finish building the tree")
        print()
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

#split all the data for k split
def split_all_data2(k, test, clients_data2, num ,clients_data):
    length = len(clients_data) * (1 - k)
    # print(length)
    for i in range(len(clients_data)):
        if(i<(length*num) and i>length*(num-1)):
            #print(i)
            test.append(i)
        else:
            clients_data2.append(i)

    # print(clients_data2)
    # print(len(clients_data2))

# Build all the tree
def build_tree2(k , num):
    decision_Tree.clear()
    branches.clear()
    last_leafs.clear()
    data_list_test.clear()

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
    clients_data22 = list()
    #print(k)
    split_all_data2(k, data_list_test2, clients_data22 , num ,clients_data)
    #print(data_list_test2)
    options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    # data_rows = list()
    # for i in clients_data2:
    # data_rows.append(i)
    nodex = get_entropy_array(options, clients_data22)
    print("The root of the tree is :", nodex.name)
    root.append(nodex.name)
    make_leafs_first(clients_data22, nodex.name, nodex.name_num, nodex.Entropydata, "start", options)
    # print("df len = ", len(decision_Tree))
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
    # print(data_list_test)
    print("Calculate the quality of the decision tree:")
    # data_list_test
    errors = 0
    good_pred = 0
    for row in data_list_test2:
        real_val = clients_data['Y'].values[row]
        def_to_check = get_list_from_client(row)
        pred_val = str(will_default2(def_to_check))
        # print(pred_val,real_val,pred_val == real_val)
        if (pred_val == real_val):
            good_pred += 1
        else:
            errors += 1
    #print(good_pred,errors)
    if(good_pred == 0 ):
        print("The Tree acc is :", 0)
        print("The Tree error is :", 1)
        return 0
    result = round((good_pred / (good_pred + errors)), 3)
    print("The Tree acc is :", result)
    print("The Tree error is :", round((errors / (good_pred + errors)), 3))
    return result

    # for row in data_list_test:
    #    def_to_check = get_list_from_client(row)
    #    will_default(def_to_check)

# need to be function with only K
def tree_error(k):
    print("Start tree_error:")
    #kfold = KFold(n_splits=k, shuffle=True, random_state=123)
    results = list()
    for i in range(k):
        print()
        print("Tree number:" , i+1)
        results.append(build_tree2(((k-1)/k) , i+1))
    total_result = 0
    for acc in results:
        total_result += acc
    print()
    print("All the tree produce")
    print("The average accuracy is :" , round ((total_result/k), 3))
    print("The average error is :" , round(((k-total_result)/k), 3))

###---------------------------- Tests -----------------------------###

#tree error check
tree_error(3)

#build tree check :
#build_tree(0.3)

# how i check will defulat :
# list of str
#list1234 = ["20000","1","2","1","32","0","0","0","2","0","0","16354","17776","21158","20511","20316","20474","1700","4000","0","800","1000","800"]
#ans = will_default(list1234)
# or list of int
#list12345 = [20000,1,2,1,32,0,0,0,2,0,0,16354,17776,21158,20511,20316,20474,1700,4000,0,800,1000,800]
#ans = will_default(list12345)

#print("the predict is : ",  ans)

