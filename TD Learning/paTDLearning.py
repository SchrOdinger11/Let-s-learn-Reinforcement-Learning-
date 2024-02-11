import copy
import numpy as np
import random
import math
import matplotlib.pyplot as plt

r=5
c=5


y=0.9
a1=0
b1=0
actions=np.array(['U','D','R','L'])



policy=[[None for _ in range(r)] for _ in range(c)]

def checkReward(i,j):
  r=0

  if(i==4 and j==2):
      r=-10

  elif (i==4 and j==4):
        r=10

  # elif (i==0 and j==2):
  #     r=5
  return r

def isTerminal(i,j):
  if i==4 and j==4 :
    return True
  return False


def isObstacle(i,j):
  if((i==2 or i==3) and (j==2)):
    return True
  return False


def transitionNorth(i,j):
  if(i>0):
    i=i-1

  if(isObstacle(i,j)):
    i=i+1
  return i,j

def transitionSouth(i,j):
  if(i<4):
    i=i+1

  if(isObstacle(i,j)):
    i=i-1
  return i,j

def transitionEast(i,j):
  if(j<4):
    j=j+1

  if(isObstacle(i,j)):
    j=j-1
  return i,j

def transitionWest(i,j):
  if(j>0):
    j=j-1
  if(isObstacle(i,j)):
    j=j+1
  return i,j


def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n




policy = np.array([
    ['R', 'R', 'R', 'D', 'D'],
    ['R', 'R', 'R', 'D', 'D'],
    ['U', 'U', None, 'D', 'D'],
    ['U', 'U', None, 'D', 'D'],
    ['U', 'U', 'R', 'R', 'GOAL']
], dtype=object)


def nextState(s0,a):


  r=0
  probabilities = [0.8, 0.05, 0.05, 0.1]  # [go in that direction, veer left, veer right, break down]

    # Sample the action based on the probabilities
  ac = np.random.choice(['go', 'left', 'right', 'break'], p=probabilities)
  i=s0[0]
  j=s0[1]
  a1=0
  b1=0
  if(a=='U'):
    if(ac=='go'):
      a1,b1=transitionNorth(i,j)

    elif (ac=='left'):
      a1,b1=transitionEast(i,j)

    elif(ac=='right'):
      a1,b1=transitionWest(i,j)
    else:
      a1=i
      b1=j
    if(isObstacle(a1,b1)):
      a1=i
      b=j
    r=checkReward(a1,b1)


  elif(a=='D'):
    if(ac=='go'):
      a1,b1=transitionSouth(i,j)

    elif(ac=='left'):
      a1,b1=transitionEast(i,j)

    elif(ac=='right'):
      a1,b1=transitionWest(i,j)

    else:
      a1=i
      b1=j

    if(isObstacle(a1,b1)):
      a1=i
      b1=j
    r=checkReward(a1,b1)


  elif(a=='R'):
    if(ac=='go'):
      a1,b1=transitionEast(i,j)

    elif(ac=='left'):
      a1,b1=transitionNorth(i,j)

    elif(ac=='right'):
      a1,b1=transitionSouth(i,j)

    else:
      a1=i
      b1=j

    if(isObstacle(a1,b1)):
      a1=i
      b1=j
    r=checkReward(a1,b1)


  elif(a=='L'):
    if(ac=='go'):
      a1,b1=transitionWest(i,j)

    elif(ac=='left'):
      a1,b1=transitionNorth(i,j)

    elif(ac=='right'):
      a1,b1=transitionSouth(i,j)

    else:
      a1=i
      b1=j

    if(isObstacle(a1,b1)):
      a1=i
      b1=j
    r=checkReward(a1,b1)

  s1=[]
  s1.append(a1)
  s1.append(b1)

  return s1,r


import numpy as np

def d0():

    grid_size = 5
    mean = 2.5
    std_dev = 1.0

    while True:

        sample = np.random.normal(loc=mean, scale=std_dev, size=2)


        sample = np.round(sample).astype(int)


        sample = np.clip(sample, 0, grid_size - 1)


        random_state = tuple(sample)


        restricted_states = [(3, 2), (2, 2),(4,4)]
        if random_state not in restricted_states:
            break

    return random_state


def generate_value_function(value_function,policy):
  c=0
  while(True):

    s0=d0()
    dell=0
    while(True):


          action=policy[s0[0]][s0[1]]
          s_new,r=nextState(s0,action)

          previous_states=value_function[s0[0]][s0[1]]



          value_function[s0[0]][s0[1]]=value_function[s0[0]][s0[1]]+alpha*(r+y*value_function[s_new[0]][s_new[1]]-value_function[s0[0]][s0[1]])




          if(s_new[0]==4 and s_new[1]==4):

            break;



          dell=max(dell, abs(previous_states-value_function[s_new[0]][s_new[1]]))
          s0=s_new



    c+=1


    if(dell<0.0001):
      epochs.append(c)
      break


  return value_function

alpha=0.6
most_optimal=[[4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
[4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
[3.8672, 4.3900, 0.0, 7.5769, 8.4637],
[3.4182, 3.8319, 0.0, 8.5738, 9.6946],
[2.9977, 2.9309, 6.0733, 9.6946, 0.0]]
sum_value_function = [[0 for _ in range(r)] for _ in range(c)]
value_function = [[0 for _ in range(r)] for _ in range(c)]
epochs=[]


for i in range (0,50):

  optimal_value_function=generate_value_function(value_function, policy)

  for i in range(r):
      for j in range(c):
            #print("{:<10}".format("{:.4f}".format(optimal_value_function[i][j])), end="")
            sum_value_function[i][j] += optimal_value_function[i][j]




print("Final Output")
for i in range(r):
    for j in range(c):
          print("{:<10}".format("{:.4f}".format(sum_value_function[i][j]/50)), end="")

    print()

print(epochs)

dell=0
for i in range(r):
  for j in range(c):
    dell=max(dell, abs(most_optimal[i][j]-sum_value_function[i][j]/50))

print("Max Norm is ",dell)

print("Average is ",np.mean(epochs))
print("Std deviation is ",np.std(epochs))


