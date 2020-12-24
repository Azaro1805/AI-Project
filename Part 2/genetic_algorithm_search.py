import math

import numpy as np
import pandas as pan
import random
import copy

# Create DB from file
def createDB (data):
    CityDB = np.array(data)
    return CityDB

# Create country DB
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

# Get a first value from set
def getVauleFromSet(set1):
    if(len(set1)==0):
        return " "
    for e in set1:
        return e

# Add ring number to the location
def addRingNum(locations , RingNumber):
    for i in range(len(countryDB)):
        city = getVauleFromSet(countryDB[i][0])
        #print("city = " , city , ", locations = " , locations, ringNumbers[i] , len(countryDB)+5)
        if(city == locations and ringNumbers[i]==len(countryDB)+5):
            citiesName[i] = city
            ringNumbers[i] = RingNumber
            #print("the location :", citiesName[i] , " , the number of ring after add :", ringNumbers[i])

# Check if  one location is in set of neighbours of other location
def myNeighborsSet(locations):
    for i in range(len(countryDB)):
        city = getVauleFromSet(countryDB[i][0])
        if(city==locations):
            return countryDB[i][1]

# Make ring number to the location and his neighbours
def makeRing(locations , ringNumber):
    NeibOfLocations = myNeighborsSet(locations)
    #print("neibset of  = ", locations, " , is: " , NeibOfLocations)
    for i in NeibOfLocations:
        addRingNum(i , ringNumber)

# Check if the end location is in the first step
def inFirstRing(endLocation):
    for i in range (len(citiesName)):
        if(citiesName[i]==endLocation):
            if(ringNumbers[i]==1):
                return True
            if (ringNumbers[i] == 0):
                return True

    return False

# Make the first ring of the start location
def StartPointRing(startLocation, endLocation):
    ringNumber = 1
    for i in range (len(countryDB)):
        country = getVauleFromSet(countryDB[i][0])
        if(country==endLocation):
            ringNumbers[i] = 1
            citiesName[i] = endLocation
            #print("the location : ", citiesName[i], ", the number of ring after add : ", ringNumbers[i])
    makeRing(endLocation, ringNumber + 1)

# Use BFS algorithm on locations
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

# Give the Heuristic Value of the location
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

# Print the name of locations from the list
def printlist (list):
    for i in list:
        print("i.name = ", i.name, ", i.dataF = ", i.dataF, ", i.next = ", i.next, ", i.before = ", i.before)

# Generate set of neighbours of the location
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

# Do all the things for the first step in the algorithm
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

# Find the lowest F value from list of city
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

# Check if location is in the list
def isInList (list , location):
    for i in list:
        if (i.name == location):
            return True
    return False

# Check if location is allready Inside Frontier with the right value
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

# Calculate DataF from each location
def caculateF (me,friend, goal_locations):
    meName = me.name
    if(allreadyInsideFrontier(me,friend) == False):
        if(friend==goal_locations):
            return me.dataF-1
        if(getHeuristicValue(me.name)==getHeuristicValue(friend)):
            return me.dataF + 0.001
        return ( me.dataF-getHeuristicValue(me.name) )+ getHeuristicValue(friend) + 0.001
    return True

# Sort list by Data F
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

# Print path
def printRouteString(pathList):
    stringPath = ""
    for i in pathList:

        stringPath = stringPath + i.name + " -> "

    stringPath = stringPath[0:len(stringPath)-4]
    #print(stringPath)

# Print path in right term
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

# Find path with A-star algorithm
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
                    F = caculateF(lowest, i , goal_locations)
                    if(F != True):
                        frontier.append(Node(i, F, myNeighborsSetCity(i), lowest.name))
                #print()
                #print("next i = ",  i)
                #printlist(fortier)
            #print("Frontier = ")
            #printlist(frontier)
            #print()

# Print array
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

# Print output if detail_ouput = True
def printOutPut2 ():
    if( pathSorted[1].name == "No path found."):
        print("No path found.")
    else:
        print("Location = ", pathSorted[1].name , " , Heuristic value = ", pathSorted[1].dataF)

# Check if route is possible in first place (connection between cities)
def NoRoute():
    k=0
    for i in countryDB:
        if(len(i[1])==0):
            for j in i[0]:
                NoRoutePossible[k]= j
                k=k+1

# A star function
def A_star_func(starting_locations,goal_locations, detail_output, arrayofPathlist, i):
    find_path_for_each_country(starting_locations, goal_locations)
    arrayofPathlist[i] = copy.deepcopy(pathSorted)
    if(detail_output):
        printOutPut2()
    pathSorted.clear()
    pathList.clear()
    frontier.clear()

    # print()

# Check if end q start city have neighbours
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

# Class Node 
class Node:
    def __init__(self,name, data , next, before):
        self.name = name
        self.dataF = data
        self.next = next
        self.before = before

#---------------------------------------- hill climbing --------------------------------------------#

def hill_climbing_search(starting_locations, goal_locations, detail_output):
    need_Restart = True
    arrayofPathlist = [list() for i in range(len(starting_locations))]
    for i in range(len(starting_locations)):
        for restart in range(5):
            print("restart = ", restart)
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
                    F = caculateF(lowest, i, goal_locations)
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

# choose city to move to in the next step
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
            lowest = choose_city_scheduleT2(detail_output)
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
                    F = caculateF(lowest, i , goal_locations)
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
def choose_city_scheduleT(detail_output3):
    t = round(getTheLowest(pathList).dataF)
    #print()
    #print("t=" ,t)
    if(t<=temperature):
        lowest = getTheLowest(frontier)
        delta = getTheLowest(pathList).dataF - lowest.dataF
        if(detail_output3):
            print(" temperature < " , temperature ,", lowest name is :", lowest.name, ", his F : ", lowest.dataF, ", last Move F : ",getTheLowest(pathList).dataF,", Diff = ", delta )
        return lowest
    lowest = getTheLowest(frontier)
    delta = getTheLowest(pathList).dataF-lowest.dataF
    if (delta>0):
        if (detail_output3):
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
                if (detail_output3):
                    print("doing this move ,probability to do this move  = ", round(prob,3)," , ", round(prob,3), ">", round(random_prob,3), "his name is :", i.name,  ", his F : ", lowest.dataF, ", last Move F : ",getTheLowest(pathList).dataF,", Diff = ", delta )
                return i
            else:
                if (detail_output3):
                    print("don't doing this move ,probability to do this move  = ", round(prob, 3), " , ", round(prob, 3), "<", round(random_prob, 3), "his name is :", i.name, ", his F : ", i.dataF,", last Move F : ", getTheLowest(pathList).dataF, ", Diff = ", delta)

# choose city to move to (from last friend neighbors only)
def choose_city_scheduleT2(detail_output):
    t = len(pathList)
    T = schedule(t)
    #print()
    #print("t=" ,t)
    if(T==0):
        lowest = getTheLowest(frontier)
        delta = pathList[len(pathList)-1].dataF - lowest.dataF
        if(detail_output):
            print(" temperature < " , temperature ,", lowest name is :", lowest.name, ", his F : ", lowest.dataF, ", last Move F : ",pathList[len(pathList)-1].dataF,", Diff = ", delta )
        return lowest
    lowest = getTheLowest(frontier)
    delta = pathList[len(pathList)-1].dataF-lowest.dataF
    if (delta>0):
        if (detail_output):
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
                if (detail_output):
                    print("doing this move ,probability to do this move  = ", round(prob,3)," , ", round(prob,3), ">", round(random_prob,3), "his name is :", i.name,  ", his F : ", i.dataF, ", last Move F : ",pathList[len(pathList)-1].dataF,", Diff = ", delta )
                return i
            else:
                if (detail_output):
                    print("don't doing this move ,probability to do this move  = ", round(prob, 3), " , ", round(prob, 3), "<", round(random_prob, 3), "his name is :", i.name, ", his F : ", i.dataF,", last Move F : ", pathList[len(pathList)-1].dataF, ", Diff = ", delta)

# schedule(t) is for generate T , T is generated linearly depending on the number of iteration.
def schedule(t):
    #print(10-(t/10))
    return 10-(t/10)

#----------------------------------------LOCAL BEAM SEARCH------------------------------------------#

def local_beam_search(starting_locations, goal_locations, detail_output):
    #need_Restart = True
    arrayofPathlist = [list() for i in range(len(starting_locations))]
    for i in range(len(starting_locations)):
        split = starting_locations[i].split(sep=", ", maxsplit=2)
        country1 = split[1]
        split = goal_locations[i].split(sep=", ", maxsplit=2)
        country2 = split[1]
        if (isInNoRoutePossible(country1, country2) == False):
            BFSCity(country1, country2)
            local_beam_func(starting_locations[i], goal_locations[i], detail_output, arrayofPathlist, i)
        else:
            arrayofPathlist[i].append(Node("No path found.", -100, set(), "No path found."))
            if (detail_output):
                print("No path found.")
        print(len(arrayofPathlist[i]))
    #printlist(arrayofPathlist[i])
    #print("atfer")
    printOutput(arrayofPathlist)

    '''if(isInList(arrayofPathlist[i] ,goal_locations[i])):
        need_Restart = False
    if (need_Restart == False):
        printOutput(arrayofPathlist)
        break'''

def local_beam_func(starting_locations, goal_locations, detail_output, arrayofPathlist,i2):
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
            open_K_opnions(goal_locations)
    # printlist(pathSorted)
    arrayofPathlist[i2] = copy.deepcopy(pathSorted)
    if (detail_output):
        printOutPut2()
    pathSorted.clear()
    pathList.clear()
    frontier.clear()

# doing K move before next section in the tree
def open_K_opnions(goal_locations):
    frontier2 = copy.deepcopy(frontier)
    for i in range (3):
        print(i)
        if(len(frontier2)==0):
            break
        lowest = getTheLowest(frontier2)

        print("the lowest name  = " ,lowest.name , ", data  = ",lowest.dataF,", before  = ", lowest.before,", set  = ", lowest.next)
        pathList.append(lowest)
        frontier2.remove(lowest)
        removeCity(lowest, frontier)
        print("frontier2 = ")
        printlist(frontier2)
        if (lowest.dataF < 1):
            break
        # if ((lowest.dataF - pathList[len(pathList) - 1].dataF) < 0.7):
        # sideMoves2 = sideMoves2 - 1
        #  print("sideMoves" , sideMoves2)
        for i in lowest.next:
            if (isInList(pathList, i) == False):
                F = caculateF(lowest, i , goal_locations)
                if (F != True):
                    frontier.append(Node(i, F, myNeighborsSetCity(i), lowest.name))
            # print("next i = ",  i)
        print("Frontier = ")
        printlist(frontier)
        print()
    print("pathList = ")
    printlist(pathList)

# remove city from list
def removeCity(lowest, list):
    for i in list:
        if (i.name == lowest.name):
            list.remove(i)


#---------------------------------------------------------------------------------------------------#

def genetic_algorithm_search(starting_locations, goal_locations, detail_output):
    arrayofPathlist = [list() for i in range(len(starting_locations))]
    for i in range(len(starting_locations)):
        split = starting_locations[i].split(sep=", ", maxsplit=2)
        country1 = split[1]
        split = goal_locations[i].split(sep=", ", maxsplit=2)
        country2 = split[1]
        if (isInNoRoutePossible(country1, country2) == False):
            BFSCity(country1, country2)
            path10 = find_Population(starting_locations[i], goal_locations[i])
            PRoute = [True for i in range(len(path10))]
            rank = get_rank(path10)
            pairs = create_pairs(rank, path10)
            childs_list = create_children(pairs, path10 , PRoute)
            create_mutation(childs_list , PRoute)
            #A_star_func(starting_locations[i], goal_locations[i], detail_output, arrayofPathlist, i)
        else:
            arrayofPathlist[i].append(Node("No path found.", -100, set(), "No path found."))
            if (detail_output):
                print("No path found.")
        #print(len(arrayofPathlist[i]))
    #if (detail_output == False):
       #printOutput(arrayofPathlist)

# Generate the population in order to create paths
def find_Population(starting_locations, goal_locations):
    firstStep(starting_locations)
    getPath = False
    f_first = pathList[0].dataF
    #print("f_first = " , f_first)
    randomCityF = random.randint(2,f_first-1)
    #print("randomCityF=" , randomCityF , "randomCityF2=" , f_first)
    listNewMGoals = list()
    find_city_in_ring(randomCityF, listNewMGoals)
    find_city_in_ring(f_first, listNewMGoals)
    print("listNewMGoals len = ", len(listNewMGoals))
    path10= [list() for i in range(10)]
    createPaths(listNewMGoals, starting_locations,goal_locations,path10)
    #printlistName(path10)
    return path10

# Insert all the location in the ring into list
def find_city_in_ring(ring , listNewMGoals):
    k=-1
    for i in (ringNumbers):
        k=k+1
        #print(cityRing)
        #print(countryDB[k][0])
        if( i == ring):
            cityRing =" "+getVauleFromSet(countryDB[k][0])
            #print("cityRing = ", cityRing)
            for j in CityDB:
                place = j[0].split(sep=",", maxsplit=2)
                #print("place=" , place)
                city = place[0]
                country = place[1]
                #print("city =" , city)
                #print("country=", country)
                place2= city+","+country
                #print(place2)
                #print(country , cityRing)
                if(country == cityRing):
                    #print("enter")
                    if (isInList2(listNewMGoals, place2) == False):
                        listNewMGoals.append(place2)
                        #print(place2)

# Check if country is in list1
def isInList2 (list1,country):
    for i in list1:
        if(i==country):
            return True
    return False

# Create 10 random Paths
def createPaths(listNewMGoals, starting_locations, goal_locations, path10):
    print("The Paths:")
    for t in range(len(path10)):
        arrayofPathlist2=[list() for i in range(2)]
        ranPlace = random.randint(0,len(listNewMGoals))
        #print("MP = " , listNewMGoals[ranPlace])
        MP=listNewMGoals[ranPlace]
        find_path_for_each_country(starting_locations, MP)
        arrayofPathlist2[0] = copy.deepcopy(pathSorted)
        #printlist(arrayofPathlist2[0])
        pathSorted.clear()
        frontier.clear()
        pathList.clear()
        firstStep(MP)
        for i in frontier :
            if(isInList(arrayofPathlist2[0] , i)):
                frontier.remove(i)
        #print("frontier")
        #printlist(frontier)
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
            if (len(frontier) == 0):
                getPath = True
                printPath("No path found.")
            if (len(pathList) > numberOfIterations):
                getPath = True
                a = ("path is more then " + (str)(numberOfIterations) + " iterations")
                printPath(a)
            if (getPath == False):
                lowest = getTheLowest(frontier)
                #print("the lowest name  = " ,lowest.name , ", data  = ",lowest.dataF,", before  = ", lowest.before,", set  = ", lowest.next)
                pathList.append(lowest)
                # print("pathList = ")
                # printlist(pathList)
                frontier.remove(lowest)
                #  print("sideMoves" , sideMoves2)
                for i in lowest.next:
                    if (isInList(pathList, i) == False and isInList(arrayofPathlist2[0], i) == False):
                        F = caculateF(lowest, i, goal_locations)
                        if (F != True):
                            frontier.append(Node(i, F, myNeighborsSetCity(i), lowest.name))
                    # print("next i = ",  i)
                # print("Frontier = ")
                # printlist(frontier)
                # print()
        # printlist(pathSorted)
        arrayofPathlist2[1] = copy.deepcopy(pathSorted)
        #printlist(arrayofPathlist2[1])
        path10[t] = bothList(arrayofPathlist2[0],arrayofPathlist2[1])
        st= str (t) +") "
        for i in path10[t]:
            if(i.name == goal_locations):
                st = st + i.name
            else:
                st = st + i.name + "-> "
        print(st)
        #printlist(pathBoth)
        pathSorted.clear()
        pathList.clear()
        frontier.clear()
        listNewMGoals.remove(listNewMGoals[ranPlace])
        #print(listNewMGoals[ranPlace])

# Generate a list from 2 list
def bothList(list1 , list2):
    list3 = copy.deepcopy(list1)
    counter1=0
    for i in (list2):
       if(counter1!=0):
           list3.append(i)
       counter1=counter1+1
    return list3

# Print Path city names
def printlistName(list):
    for i in list:
        path = ""
        for j in i:
            path = path + j.name + " -> "
        print(path)

# Generate to each path rank
def get_rank (all_path):
    rank = [ 0 for i in range (len(all_path))]
    j=0
    min = len(all_path[0])
    total = 0
    for i in all_path:
        rank[j] = len(i)
        if(len(i)<min):
            min=len(i)
        #print(rank[j])
        j=j+1
    j = 0
    for k in rank:
        rank[j] = round(min/rank[j],3)
        total = total+rank[j]
        #print(rank[j])
        j=j+1
    j = 0
    for k in rank:
        rank[j] = round(rank[j]/total,3)
        #print(rank[j])
        j=j+1
    return rank

# Create pairs from 10 path and their ranks
def create_pairs(rank,path10):
    print()
    print("the pairs is:")
    pairs = [-1 for i in range (len(path10)*2)]
    pairs2 = [-1 for i in range (2)]
    t=0

    for m in range( len(path10)):
        con2 =True
        while(con2):
            for i in range (2):
                j=0
                rn = random.random()
                con=True
                prob = rank[0]
                #print(i)
                while(con):
                    if(i==0): # i=0
                        if(rn < prob):
                            pairs2[0]= j
                            i=i+1
                            con = False
                        else:
                            j=j+1
                            prob = prob + rank[j]
                    else: # i=1
                        if (rn < prob):
                            if(pairs2[0]!=j):
                                pairs2[1] = j
                                con = False
                            else:
                                j = j + 1
                                if (j == len(rank)):
                                    rn = random.random()
                                    j = 0
                                else:
                                    prob = prob + rank[j]
                        else:
                            j = j + 1
                            if (j == len(rank)):
                                rn = random.random()
                                j = 0
                            else:
                                prob = prob + rank[j]
            if(dup_pair(pairs2[0] , pairs2[1] , pairs) == False):
               con2 = False
            else:
                pairs2[0] = -1
                pairs2[1] = -1
        for i in pairs2:
            pairs[t]=i
            t=t+1

    for i in range(0,len(pairs),2) :
        print("[", i , "]","[", i+1 , "]")
    return pairs

# Check if their is a duplicate pair in pair list
def dup_pair(a , b , pairs):
    for i in range(len(pairs)) :
        if(i%2==0):
            if(a==pairs[i] and b==pairs[i+1]):
                #print(a ,b, pairs[i],pairs[i+1])
                return True
    return False

# Generate children from the 10 paths and the pairs i create
def create_children(pairs , path10, PRoute):
    childs_list = [list() for i in range(len(path10))]
    i2=0
    for i in range(0,len(pairs),2):
        path1 = path10[pairs[i]]
        path2 =  path10[pairs[i+1]]
        path3 = list()
        if(len(path1)>=len(path2)):
            runNum = random.randint(1, len(path2)-1)
        else:
            runNum = random.randint(1, len(path1)-1)
        place1 = path1[runNum-1]
        place2 = path2[runNum]
        k=0
        for j in path1:
            if(k==runNum):
                break
            path3.append(j)
            k=k+1
        k=0
        for l in path2:
            if(k>=runNum):
                path3.append(l)
            k=k+1
        for m in path3:
            childs_list[i2].append(m) # not in 0
        st =""
        for i3 in childs_list[i2]:
            st = st + i3.name + "-> "

        print()
        print("child " , i2 , " :")
        print(st)
        print("from which node i cut  = " , runNum)
        check_the_connections(path3, runNum, place1, place2, PRoute, i2)
        i2= i2+1
    return childs_list

# Check if place1 neighbor of place2
def check_the_connections(path3, cutNum, place1, place2 , PRoute, i2):
    print("place of connections : ", place1.name, "  ,  ", place2.name)
    for i in place1.next:
        if(i == place2.name):
            print("they neighbor, possible path")
            return True
    PRoute[i2] = False
    print("they not neighbors, not possible path")
    return False

# Change one node randomly in the list of each path
def create_mutation(childs_list, PRoute):
    for i in range(len(childs_list)):
        print()
        path1=childs_list[i]
        ranNum = random.randint(1,len(path1)-2)
        print("The Child : ", i ," The mutation is in Node: ", ranNum)
        placeBefore = path1[ranNum - 1]
        print("placeBefore", placeBefore.name)
        placeAfter = path1[ranNum + 1]
        print("placeAfter", placeAfter.name)
        flag = False
        print("Path before change : ")
        st=""
        for j in childs_list[i]:
            st = st+ j.name+"-> "
        print(st)
        for j in childs_list[i]:
            if (j.name == path1[ranNum].name):
                break
        childs_list[i].remove(path1[ranNum])
        t=0
        lenset = len(placeBefore.next)
        flag = False
        for j in placeBefore.next:
            setNeib =myNeighborsSetCity(j)
            if(is_in_set(setNeib , placeAfter)):
                childs_list[i].insert(ranNum,Node(j, placeBefore.dataF+0.001, setNeib, placeBefore.name))
                flag = True
                break
            if(t==lenset-1):
                childs_list[i].insert(ranNum,Node(j, placeBefore.dataF+0.001, setNeib, placeBefore.name))
            t=t+1
        print("Path before change : ")
        st = ""
        for j in childs_list[i]:
            st = st+ j.name+"-> "
        print(st)
        if(flag):
            if(PRoute[i]):
                print("the mutation have connections with nodes, possible path")
            else:
                print("the mutation have connections with nodes, but the path is not possible")
        else:
            print("the mutation haven't connections with nodes, no possible path")

# Check if location is in set
def is_in_set(set1 , location):
    if(len(set1)==0):
        return False
    for i in set1:
        if(i==location.name):
            return True
    return False

#-------------------------------------- find_path function -------------------------------------------------#


def find_path(starting_locations, goal_locations, search_method, detail_output):
    frontier.clear()
    pathList.clear()
    if (search_method == 1):
        A_star_search(starting_locations, goal_locations, detail_output)
    if (search_method == 2):
        hill_climbing_search(starting_locations, goal_locations, detail_output)
    if (search_method == 3):
        simulated_annealing_search(starting_locations, goal_locations, detail_output)
    if (search_method == 4):
        local_beam_search(starting_locations, goal_locations, detail_output)
    if (search_method == 5):
        genetic_algorithm_search(starting_locations, goal_locations, detail_output)

#----------------------------------------------------------------------------------------------------------#



print()
print("Start the Program")
print()

# the path to the CSV file of locations  (adjacency file)
data = pan.read_csv(r'C:\Users\oraza\Downloads\adjacency.csv')

CityDB = createDB(data)
#print(CityDB)
countryDB = createCountryDB()
#print(countryDB)
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
#print()
#find_path(startList, endList, 2, False)
#print()
find_path(startList, endList, 5, False)
print()
print("End the Program")
