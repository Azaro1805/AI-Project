import math

import numpy as np
import pandas as pan
import random
import copy

from numpy.testing._private.parameterized import param


def createDB (data):
    CityDB = np.array(data)
    return CityDB

def createCountryDB ():
    CountryDB = [[set() for j in range(2)] for i in range(58)]
    k = 0
    for i in (CityDB):
        alreadyInside = False
        Country = i[0].split(sep=", ", maxsplit=2)
        for j in range(len(CountryDB)):
            Country2 = getVauleFromSet(CountryDB[j][0])
            if (Country2 == Country[1]):
                alreadyInside = True
                break
        if (alreadyInside == False):
            CountryDB[k][0].add(Country[1])
            for n in CityDB:
                alreadyInside2 = False
                fromCountry = n[0].split(sep=", ", maxsplit=2)
                toCountry = n[1].split(sep=", ", maxsplit=2)
                if (Country[1] == fromCountry[1]):
                    if(Country[1] == toCountry[1]):
                        alreadyInside2 = True
                    for t in CountryDB[k][1]:
                        if (toCountry[1] == t):
                            alreadyInside2 = True
                            break
                    if (alreadyInside2 == False):
                        CountryDB[k][1].add(toCountry[1])
            k = k + 1
    return CountryDB

def getVauleFromSet(set1):
    if(len(set1)==0):
        return " "
    for e in set1:
        return e

def addRingNum(locations , RingNumber):
    for i in range(len(countryDB)):
        city = getVauleFromSet(countryDB[i][0])
        #print("city = " , city , ", locations = " , locations, ringNumbers[i] , len(countryDB)+5)
        if(city == locations and ringNumbers[i]==len(countryDB)+5):
            citiesName[i] = city
            ringNumbers[i] = RingNumber
            #print("the location :", citiesName[i] , " , the number of ring after add :", ringNumbers[i])

def myNeighborsSet(locations):
    for i in range(len(countryDB)):
        city = getVauleFromSet(countryDB[i][0])
        if(city==locations):
            return countryDB[i][1]

def makeRing(locations , ringNumber):
    NeibOfLocations = myNeighborsSet(locations)
    #print("neibset of  = ", locations, " , is: " , NeibOfLocations)
    for i in NeibOfLocations:
        addRingNum(i , ringNumber)

def inFirstRing(endLocation):
    for i in range (len(citiesName)):
        if(citiesName[i]==endLocation):
            if(ringNumbers[i]==1):
                return True
            if (ringNumbers[i] == 0):
                return True

    return False

def StartPointRing(startLocation, endLocation):
    ringNumber = 1
    for i in range (len(countryDB)):
        country = getVauleFromSet(countryDB[i][0])
        if(country==endLocation):
            ringNumbers[i] = 1
            citiesName[i] = endLocation
            #print("the location : ", citiesName[i], ", the number of ring after add : ", ringNumbers[i])
    makeRing(endLocation, ringNumber + 1)

def BFSCity(startLocation, endLocation):
    arriveToEnd = False
    ringNumber = 1
    for i in range(len(ringNumbers)):
        ringNumbers[i] = len(countryDB) + 5
        citiesName[i] = ""
    StartPointRing(startLocation, endLocation)
    #print(citiesName)
    #print(ringNumbers)
    arriveToEnd=inFirstRing(startLocation)
    #print(arriveToEnd)
    while (arriveToEnd == False):
        ringNumber = ringNumber + 1
        #print("flag = ", arriveToEnd, ", RingNumber = ", ringNumber)
        for i in range(len(countryDB)):
            if (arriveToEnd == True):
                break
            # print("countryDB[i][0] =" , countryDB[i][0] , ", ringNumber = " , ringNumbers[i])
            if (ringNumbers[i] == ringNumber):
                # print("neibSet befor = ", myNeighborsSet(neighborsRing[i][0]))
                # neibSet = myNeighborsSet(neighborsRing[i][0])
                ringNumber2 = ringNumber + 1
                # print("neighborsRing = ", np.matrix(neighborsRing))
                #print("###################################### new firends of the ring  : #######################################")
                #print(" name  = ", countryDB[i][0], "RingNumber = ", ringNumbers[i], ", new ring =", ringNumber2)
                city2 = getVauleFromSet(countryDB[i][0])
                makeRing(city2, ringNumber2)
                if (city2 == startLocation):
                    #print("%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    #print("i = endLocation =", city2)
                    arriveToEnd = True
                    break
    #print("neighborsRing = ", citiesName)
    #print("neighborsRing = ", ringNumbers)

def getHeuristicValue(location):
    locationSplit = location.split(sep=", ", maxsplit=2)
    city = locationSplit[0]
    Country = locationSplit[1]
    #print("Country = ", Country ,", City = ", city)
    for i in range (len(citiesName)):
        Country2 = getVauleFromSet(countryDB[i][0])
        #print("Country2 = ",Country2, " , Country=" , Country)
        if(Country2 == Country):
            return ringNumbers[i]
    return 61

def printlist (list):
    for i in list:
        print("i.name = ", i.name, ", i.dataF = ", i.dataF, ", i.next = ", i.next, ", i.before = ", i.before)

def myNeighborsSetCity (location):
    set1 = set()
    for i in (CityDB):
        fromCountry = i[0].split(sep=", ", maxsplit=2)
        toCountry = i[1].split(sep=", ", maxsplit=2)
        city = fromCountry[0]+", "+fromCountry[1]
        city2 = toCountry[0]+", "+toCountry[1]
        if (city == location and city2!= location):
            set1.add(city2)
    return set1

def firstStep(starting_locations):
    set = myNeighborsSetCity(starting_locations)
    pathList.append(Node(starting_locations,getHeuristicValue(starting_locations),set,"Start Point"))
    #print()
    #print(set)
    for i in set:
        frontier.append(Node(i, getHeuristicValue(i) + 0.001, myNeighborsSetCity(i), starting_locations))
    #print("frontier after first step")
    #printlist(frontier)
    #print("path after first step")
    #printlist(pathList)
    #print()

def getTheLowest (list):
    min = 65
    counter = 1
    name = ""
    for i in list:
        if (counter==1):
            min = i.dataF
            name = i
        if(min>i.dataF):
            min = i.dataF
            name = i
        counter = counter+1
    return name

def isInList (list , location):
    for i in list:
        if (i.name == location):
            return True
    return False

def allreadyInsideFrontier(me,friend):
    for i in frontier:
        if (i.name == friend):
            #print(i.name , " , i.DatafF = " , i.dataF, " , i.before = " , i.before)
            if (getHeuristicValue(me.name) == getHeuristicValue(friend)):
                if (i.dataF > (me.dataF + 0.001)):
                    i.dataF = me.dataF + 0.001
                    i.before = me.name
            else:
                newF = (me.dataF - getHeuristicValue(me.name)) + getHeuristicValue(friend) + 0.001
                if (i.dataF > newF):
                    i.dataF = newF
                    i.before = me.name
            #print(i.name , " , i.DatafF = " , i.dataF, " , i.before = " , i.before)
            return True
    return False

def caculateF (me,friend):
    meName = me.name
    if(allreadyInsideFrontier(me,friend) == False):
        if(getHeuristicValue(me.name)==getHeuristicValue(friend)):
            return me.dataF + 0.001
        return ( me.dataF-getHeuristicValue(me.name) )+ getHeuristicValue(friend) + 0.001
    return True

def sortNodeList(nodeList):
    #printlist(nodeList)
    #print()
    sorted = list()
    max = 0
    max2 = 0
    finish =False
    while (finish == False):
        max = 0
        max2 = 0
        for i in nodeList:
            if (i.dataF > max):
                max = i.dataF
                max2 = i.dataF
            if (i.dataF < max and (max- i.dataF)<0.2):
                if(i.dataF<max2):
                    max2= i.dataF

        #print("max = " , max , " , max2 = " , max2)
        for j in nodeList:
            if (j.dataF == max2):
                sorted.append(j)
                nodeList.remove(j)
                if(len(nodeList)==0):
                    finish = True
                    break


    #printlist(sorted)
    return sorted

def printRouteString(pathList):
    stringPath = ""
    for i in pathList:

        stringPath = stringPath + i.name + " -> "

    stringPath = stringPath[0:len(stringPath)-4]
    #print(stringPath)

def printPath(endLocation):
    path = list()
    Finish = False
    lastLocation = ""
    endLocationInside = False
    noPath = False
    if("No path found, too many sideMoves" == endLocation):
        noPath= True
        pathSorted.append(Node("No path found, too many sideMoves", -100, set(), "No path found."))
        pathSorted.append(Node("No path found, too many sideMoves", -100, set(), "No path found."))
    if("No path found." == endLocation):
        noPath = True
        pathSorted.append(Node("No path found.", -100, set(), "No path found."))
        pathSorted.append(Node("No path found.", -100, set(), "No path found."))
    if ("path is more then " + (str) (numberOfIterations) + " iterations" == endLocation):
        noPath = True
        pathSorted.append(Node("path is more then " + (str) (numberOfIterations) + " iterations", -100, set(), "No path found."))
        pathSorted.append(Node("path is more then " + (str) (numberOfIterations) + " iterations", -100, set(), "No path found."))
    if(noPath == False):
        while (Finish == False):
            for i in pathList :
                if(endLocation == i.name and endLocationInside==False):
                    #print("End location")
                    #print(i.name, " , i.DatafF = ", i.dataF, " , i.before = ", i.before)
                    path.append(i)
                    lastLocation = i.before
                    endLocationInside = True
                    #print("lastLocation = ", lastLocation)
                if (lastLocation == i.name ):
                    #print(i.name, " , i.DatafF = ", i.dataF, " , i.before = ", i.before)
                    path.append(i)
                    lastLocation = i.before
                    #print("lastLocation = ", lastLocation)
                    if (lastLocation == "Start Point"):
                        Finish = True
                        break
        path=sortNodeList(path)
        printRouteString(path)

        for i in path:
            pathSorted.append(i)

def createOutPutPrint():
    print("A")

def find_path_for_each_country(starting_locations, goal_locations):

    firstStep(starting_locations)
    getPath = False

    while (getPath == False):
        if (isInList(pathList, goal_locations) == True):
            getPath = True
            #print("##################### FIND PATH ####################")
            printPath(goal_locations)
            #print(" frontier")
            #printlist(frontier)
            break
        #print(len(frontier))
        if (len(frontier) == 0):
            getPath = True
            printPath("No path found.")
        if(len(pathList) > numberOfIterations):
            getPath = True
            a = ("path is more then " + (str) (numberOfIterations) + " iterations")
            printPath(a)
        if(getPath==False):
            lowest = getTheLowest(frontier)
            #print("the lowest name  = " ,lowest.name , ", data  = ",lowest.dataF,", before  = ", lowest.before,", set  = ", lowest.next)
            pathList.append(lowest)
            #print("pathList = ")
            #printlist(pathList)

            frontier.remove(lowest)

            for i in lowest.next:
                if (isInList(pathList,i) == False):
                    F = caculateF(lowest, i)
                    if(F != True):
                        frontier.append(Node(i, F, myNeighborsSetCity(i), lowest.name))
                #print()
                #print("next i = ",  i)
                #printlist(fortier)
            #print("Frontier = ")
            #printlist(frontier)
            #print()

def printOutput(array):
        max = len(array[0])
        for i in array:
            if(max < len(i)):
                max = len(i)

        pathArray = [["" for j in range (len(array))] for i in range(max)]
        for j in range(len(pathArray[0])):
            list1 = array[j]
            for i in range (len(pathArray)):
                if (len(array[j])-1==i):
                    pathArray[i][j] = list1[i].name
                if(len(array[j])-1>i):
                    pathArray[i][j] = list1[i].name
                if (pathArray[i][j]==""):
                    pathArray[i][j]=pathArray[i-1][j]

        output = ["" for i in range (len(pathArray))]
        for i in range(len(output)):
            output[i] = "{"
            for j in range(len(pathArray[0])):
                output[i]  = output[i] + pathArray[i][j] + " ; "
            output[i] = output[i][0:(len(output[i])-3)]+"}"
            print(output[i])

def printOutPut2 ():
    if( pathSorted[1].name == "No path found."):
        print("No path found.")
    else:
        print("Location = ", pathSorted[1].name , " , Heuristic value = ", pathSorted[1].dataF)

def NoRoute():
    k=0
    for i in countryDB:
        if(len(i[1])==0):
            for j in i[0]:
                NoRoutePossible[k]= j
                k=k+1

def A_star_func(starting_locations,goal_locations, detail_output, arrayofPathlist, i):
    find_path_for_each_country(starting_locations, goal_locations)
    arrayofPathlist[i] = copy.deepcopy(pathSorted)
    if(detail_output):
        printOutPut2()
    pathSorted.clear()
    pathList.clear()
    frontier.clear()

    # print()

def isInNoRoutePossible(location,location1):
    for i in range (len(NoRoutePossible)):
        if (location == NoRoutePossible[i] and location != location1):
            return True
        if (location1 == NoRoutePossible[i] and location != location1):
            return True
    return False

def A_star_search(starting_locations, goal_locations, detail_output):
    arrayofPathlist = [list() for i in range(len(starting_locations))]
    for i in range(len(starting_locations)):
        split = starting_locations[i].split(sep=", ", maxsplit=2)
        country1 = split[1]
        split = goal_locations[i].split(sep=", ", maxsplit=2)
        country2 = split[1]
        if (isInNoRoutePossible(country1, country2) == False):
            BFSCity(country1, country2)
            A_star_func(starting_locations[i], goal_locations[i], detail_output, arrayofPathlist, i)
        else:
            arrayofPathlist[i].append(Node("No path found.", -100, set(), "No path found."))
            if (detail_output):
                print("No path found.")
        print(len(arrayofPathlist[i]))
    if (detail_output == False):
        printOutput(arrayofPathlist)

class Node:
    def __init__(self,name, data , next, before):
        self.name = name
        self.dataF = data
        self.next = next
        self.before = before

#---------------------------------------- hill climbing --------------------------------------------#

def hill_climbing_search(starting_locations, goal_locations, detail_output):
    need_Restart = True
    for i in range (5):
        arrayofPathlist = [list() for i in range(len(starting_locations))]
        for i in range(len(starting_locations)):
            split = starting_locations[i].split(sep=", ", maxsplit=2)
            country1 = split[1]
            split = goal_locations[i].split(sep=", ", maxsplit=2)
            country2 = split[1]
            if (isInNoRoutePossible(country1, country2) == False):
                BFSCity(country1, country2)
                hill_climbing_func(starting_locations[i], goal_locations[i], detail_output, arrayofPathlist, i)
            else:
                arrayofPathlist[i].append(Node("No path found.", -100, set(), "No path found."))
                if (detail_output):
                    print("No path found.")
            print(len(arrayofPathlist[i]))
        #printlist(arrayofPathlist[i])
        #print("atfer")
        if(isInList(arrayofPathlist[i] ,goal_locations[i])):
            need_Restart = False
        if (need_Restart == False):
            printOutput(arrayofPathlist)
            break

def hill_climbing_func(starting_locations, goal_locations, detail_output, arrayofPathlist, i2):
    sideMoves2 = sideMoves
    firstStep(starting_locations)
    #printlist(frontier)
    #print("path list :")
    #printlist(pathList)
    getPath = False

    while (getPath == False):
        if (isInList(pathList, goal_locations) == True):
            getPath = True
            print("sideMoves2 done " , sideMoves- sideMoves2)
            #print("##################### FIND PATH ####################")
            printPath(goal_locations)
            # print(" frontier")
            # printlist(frontier)
            break
        # print(len(frontier))
        if(sideMoves2 == 0):
            getPath = True
            printPath("No path found, too many sideMoves")
        if (len(frontier) == 0):
            getPath = True
            printPath("No path found.")
        if (len(pathList) > numberOfIterations):
            getPath = True
            a = ("path is more then " + (str)(numberOfIterations) + " iterations")
            printPath(a)
        if (getPath == False):
            lowest = choose_city()
            #print("the lowest name  = " ,lowest.name , ", data  = ",lowest.dataF,", before  = ", lowest.before,", set  = ", lowest.next)
            pathList.append(lowest)
            #print("pathList = ")
            #printlist(pathList)

            frontier.remove(lowest)
            if((lowest.dataF-pathList[len(pathList)-1].dataF)<0.7):
                sideMoves2 = sideMoves2-1
              #  print("sideMoves" , sideMoves2)
            for i in lowest.next:
                if (isInList(pathList, i) == False):
                    F = caculateF(lowest, i)
                    if (F != True):
                        frontier.append(Node(i, F, myNeighborsSetCity(i), lowest.name))
                #print("next i = ",  i)
            #print("Frontier = ")
            #printlist(frontier)
            #print()
    #printlist(pathSorted)
    arrayofPathlist[i2] = copy.deepcopy(pathSorted)
    if (detail_output):
        printOutPut2()
    pathSorted.clear()
    pathList.clear()
    frontier.clear()

def choose_city():
    min = 6500
    counter = 1
    name = ""
    city_list = list()
    for i in frontier:
        if (counter == 1):
            min = i.dataF
            name = i
        if (min >= i.dataF):
            city_list.append(i)
            min = i.dataF
            name = i
        counter = counter + 1

    exit = False
    while (exit == False):
        counter=0
        for i in city_list:
            if (i.dataF != min):
                city_list.remove(i)
                counter = counter+1
        if(counter==0):
            exit = True

    #printlist(city_list)

    if(len(city_list)==1):
        return name

    random_num = random.randint(1,len(city_list))
    #print(random_num)
    counter=1
    for i in city_list:
        if (counter==random_num):
            return i
        counter=counter+1

#------------------------------------- simulated_annealing --------------------------------------#

def simulated_annealing_search(starting_locations, goal_locations, detail_output):
    arrayofPathlist = [list() for i in range(2)]
    for i in range(len(starting_locations)):
        split = starting_locations[i].split(sep=", ", maxsplit=2)
        country1 = split[1]
        split = goal_locations[i].split(sep=", ", maxsplit=2)
        country2 = split[1]
        if (isInNoRoutePossible(country1, country2) == False):
            BFSCity(country1, country2)
            simulated_annealing_func(starting_locations[i], goal_locations[i], detail_output, arrayofPathlist, i)
        else:
            arrayofPathlist[i].append(Node("No path found.", -100, set(), "No path found."))
            if (detail_output):
                print("No path found.")
        print(len(arrayofPathlist[i]))
    #if (detail_output == False):
        printOutput(arrayofPathlist)

def simulated_annealing_func(starting_locations, goal_locations, detail_output, arrayofPathlist, i2):
    #sideMoves2 = sideMoves

    firstStep(starting_locations)
    # printlist(frontier)
    # print("path list :")
    # printlist(pathList)
    getPath = False

    while (getPath == False):
        if (isInList(pathList, goal_locations) == True):
            getPath = True
            #print("sideMoves2 done ", sideMoves - sideMoves2)
            # print("##################### FIND PATH ####################")
            printPath(goal_locations)
            # print(" frontier")
            # printlist(frontier)
            break
        # print(len(frontier))
        #if (sideMoves2 == 0):
        #    getPath = True
        #    printPath("No path found, too many sideMoves")
        if (len(frontier) == 0):
            getPath = True
            printPath("No path found.")
        if (len(pathList) > numberOfIterations):
            getPath = True
            a = ("path is more then " + (str)(numberOfIterations) + " iterations")
            printPath(a)
        if (getPath == False):
            lowest = choose_city_scheduleT2()
            #print("end2")
            #print("the lowest name  = " ,lowest.name , ", data  = ",lowest.dataF,", before  = ", lowest.before,", set  = ", lowest.next)
            pathList.append(lowest)
            #print("pathList = ")
            #printlist(pathList)

            frontier.remove(lowest)
            #if ((lowest.dataF - pathList[len(pathList) - 1].dataF) < 0.7):
            #    sideMoves2 = sideMoves2 - 1
            #  print("sideMoves" , sideMoves2)

            #if choose_city_scheduleT2 (down unil only)  :
            frontier.clear()

            #only ^^^^^^^^

            for i in lowest.next:
                if (isInList(pathList, i) == False):
                    F = caculateF(lowest, i)
                    if (F != True):
                        frontier.append(Node(i, F, myNeighborsSetCity(i), lowest.name))
                # print("next i = ",  i)
            #print("Frontier = ")
            #printlist(frontier)
            #print()
    # printlist(pathSorted)
    arrayofPathlist[i2] = copy.deepcopy(pathSorted)
    if (detail_output):
        printOutPut2()
    pathSorted.clear()
    pathList.clear()
    frontier.clear()

# choose city to move to (from all friend neighbors)
def choose_city_scheduleT():
    t = round(getTheLowest(pathList).dataF)
    #print()
    #print("t=" ,t)
    if(t<=temperature):
        lowest = getTheLowest(frontier)
        delta = getTheLowest(pathList).dataF - lowest.dataF
        print(" temperature < " , temperature ,", lowest name is :", lowest.name, ", his F : ", lowest.dataF, ", last Move F : ",getTheLowest(pathList).dataF,", Diff = ", delta )
        return lowest
    lowest = getTheLowest(frontier)
    delta = getTheLowest(pathList).dataF-lowest.dataF
    if (delta>0):
            print("Find better move , lowest name is :", lowest.name, ", his F : ", lowest.dataF, ", last Move F : ",getTheLowest(pathList).dataF,", Diff = ", delta)
            return lowest
    find_city = False
    while(find_city==False):
        for i in frontier:
            random_prob = random.random()
            delta = getTheLowest(pathList).dataF - i.dataF
            prob = math.exp(delta / t)
            #print("dont doing this move , lowest.data =",  getTheLowest(pathList).dataF,", i.dataF" , i.dataF, ", delta = ",delta, ", t = " , t," , delta / t =  ", delta / t ,", prob = ", prob)
            if (prob > random_prob):
                print("doing this move ,probability to do this move  = ", round(prob,3)," , ", round(prob,3), ">", round(random_prob,3), "his name is :", i.name,  ", his F : ", lowest.dataF, ", last Move F : ",getTheLowest(pathList).dataF,", Diff = ", delta )
                return i
            else:
                print("don't doing this move ,probability to do this move  = ", round(prob, 3), " , ", round(prob, 3), "<", round(random_prob, 3), "his name is :", i.name, ", his F : ", i.dataF,", last Move F : ", getTheLowest(pathList).dataF, ", Diff = ", delta)

# choose city to move to (from last friend neighbors only)
def choose_city_scheduleT2():
    t = len(pathList)
    T = schedule(t)
    #print()
    #print("t=" ,t)
    if(T==0):
        lowest = getTheLowest(frontier)
        delta = pathList[len(pathList)-1].dataF - lowest.dataF
        print(" temperature < " , temperature ,", lowest name is :", lowest.name, ", his F : ", lowest.dataF, ", last Move F : ",pathList[len(pathList)-1].dataF,", Diff = ", delta )
        return lowest
    lowest = getTheLowest(frontier)
    delta = pathList[len(pathList)-1].dataF-lowest.dataF
    if (delta>0):
            print("Find better move , lowest name is :", lowest.name, ", his F : ", lowest.dataF, ", last Move F : ",pathList[len(pathList)-1].dataF,", Diff = ", delta)
            return lowest
    find_city = False
    while(find_city==False):
        for i in frontier:
            random_prob = random.random()
            delta = pathList[len(pathList)-1].dataF - i.dataF
            #print(pathList[len(pathList)-1].dataF , i.dataF)
            #print(delta)

            prob = math.exp(delta / T)
            if (prob > random_prob):
                print("doing this move ,probability to do this move  = ", round(prob,3)," , ", round(prob,3), ">", round(random_prob,3), "his name is :", i.name,  ", his F : ", i.dataF, ", last Move F : ",pathList[len(pathList)-1].dataF,", Diff = ", delta )
                return i
            else:
                print("don't doing this move ,probability to do this move  = ", round(prob, 3), " , ", round(prob, 3), "<", round(random_prob, 3), "his name is :", i.name, ", his F : ", i.dataF,", last Move F : ", pathList[len(pathList)-1].dataF, ", Diff = ", delta)

# schedule(t) is for generate T , T is generated linearly depending on the number of iteration.
def schedule(t):
    #print(10-(t/10))
    return 10-(t/10)


#---------------------------------------------------------------------------------------------------#

def find_path(starting_locations, goal_locations, search_method, detail_output):
    if (search_method == 1):
        A_star_search(starting_locations, goal_locations, detail_output)
    if (search_method == 2):
        hill_climbing_search(starting_locations, goal_locations, detail_output)
    if (search_method == 3):
        simulated_annealing_search(starting_locations, goal_locations, detail_output)
    if (search_method == 4):
        print("A local beam search,")
    if (search_method == 5):
        print("A genetic algorithm.")

#---------------------------------------------------------------------------------------------------#



print()
print("Start the Algorithm")
print()

# the path to the CSV file of locations  (adjacency file)
data = pan.read_csv(r'C:\Users\oraza\Downloads\adjacency.csv')

CityDB = createDB(data)
countryDB = createCountryDB()
ringNumbers = [len(countryDB) + 5 for i in range(len(countryDB))]
citiesName = ["" for i in range(len(countryDB))]
NoRoutePossible = ["" for i in range(len(countryDB))]
NoRoute()
pathList = list()
pathSorted = list()
frontier = list()

# here i decide Parameters the Algorithm can run in order to find a path:

numberOfIterations = 750
sideMoves = 20
temperature = 1

# example how i test it with array or list of starting and ending locations:

startList = [" " for i in range(1)]
endList = [" " for i in range(1)]

#startList[0]="Curry County, OR"
#startList[0]="Fairfield County, CT"
startList[0]="Washington County, UT"

#startList[1]="Chicot County, AR"

#endList[0]="Rensselaer County, NY"
endList[0]="San Diego County, CA"

#endList[1]="Chicot County, AR"

'''startList = list()
endList = list()
startList.append("Washington County, UT")
startList.append("Chicot County, AR")
endList.append("San Diego County, CA")
endList.append("Bienville Parish, LA")'''
#find_path(startList, endList, "A*", False)
find_path(startList, endList, 1, False)
print()
find_path(startList, endList, 3, False)
print()
print("End the Algorithm")
