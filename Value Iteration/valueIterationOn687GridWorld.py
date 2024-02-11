import copy
import numpy as np
import random
import math
import matplotlib.pyplot as plt

r=5
c=5


y=0.9 

actions=np.array(['U','D','R','L'])

value_function = [[0 for _ in range(r)] for _ in range(c)]

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

def max_action_value_1(value_function,y,i,j,action ):
  rewardFromAction=[]
  val=0
  for a in action:
    if(a=='U'):
      a,b=transitionNorth(i,j)
      val=val+0.8*(checkReward(a,b)+y*value_function[a][b])

      c,d=transitionEast(i,j)
      val=val+0.05*(checkReward(c,d)+y*value_function[c][d])

      e,f=transitionWest(i,j)
      val=val+0.05*(checkReward(e,f)+y*value_function[e][f])

      val=val+0.1*(checkReward(i,j)+y*value_function[i][j])

      rewardFromAction.append(val)
      val=0

    elif(a=='D'):
      a,b=transitionSouth(i,j)
      val=val+0.8*(checkReward(a,b)+y*value_function[a][b])
      c,d=transitionEast(i,j)
      val=val+0.05*(checkReward(c,d)+y*value_function[c][d])

      e,f=transitionWest(i,j)
      val=val+0.05*(checkReward(e,f)+y*value_function[e][f])

      val=val+0.1*(checkReward(i,j)+y*value_function[i][j])

      rewardFromAction.append(val)
      val=0


    elif(a=='R'):
      a,b=transitionEast(i,j)
      val=val+0.8*(checkReward(a,b)+y*value_function[a][b])

      c,d=transitionNorth(i,j)
      val=val+0.05*(checkReward(c,d)+y*value_function[c][d])

      e,f=transitionSouth(i,j)
      val=val+0.05*(checkReward(e,f)+y*value_function[e][f])

      val=val+0.1*(checkReward(i,j)+y*value_function[i][j])

      rewardFromAction.append(val)
      val=0


    elif(a=='L'):
      a,b=transitionWest(i,j)
      val=val+0.8*(checkReward(a,b)+y*value_function[a][b])

      c,d=transitionNorth(i,j)
      val=val+0.05*(checkReward(c,d)+y*value_function[c][d])

      e,f=transitionSouth(i,j)
      val=val+0.05*(checkReward(e,f)+y*value_function[e][f])

      val=val+0.1*(checkReward(i,j)+y*value_function[i][j])

      rewardFromAction.append(val)
      val=0
    result = truncate(max(rewardFromAction), 4)

  return max(rewardFromAction), np.argmax(rewardFromAction)
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






def generate_value_function(value_function,policy):


  counter=0
  while(True):

    dell=0
    for i in range (0, r):
      for j in range (0,c):

        if not(i==4 and j==4) and not(i==2 and j==2) and not(i==3 and j==2):#and not(i==0 and j==2):
          previous_states=value_function[i][j]
          value_function[i][j],action=max_action_value_1(value_function,y,i,j,actions)


          if(action==0):
            action='U'
          elif (action == 1):
            action ='D'
          elif (action ==2):
            action ='R'
          else:
            action = 'L'
          policy[i][j]=action



          dell=max(dell, abs(previous_states-value_function[i][j]))


    
    counter+=1
    print(counter)

    if(dell<0.0001):

      break


  return value_function


optimal_value_function=generate_value_function(value_function, policy)
print("Value Function")
for i in range(r):
    for j in range(c):
     


         
           print("{:<10}".format("{:.4f}".format(optimal_value_function[i][j])), end="")
   


    print()

direction_mapping = {'R': '→', 'D': '↓', 'U': '↑', 'L': '←', None: ' ','*':'G'}
policy[4][4]='*'
#policy[0][2]='*'

print()
print()
print("Optimal Policy")
for row in policy:
    for direction in row:


       print(f"{direction_mapping[direction]:^4}", end="")
    print()


