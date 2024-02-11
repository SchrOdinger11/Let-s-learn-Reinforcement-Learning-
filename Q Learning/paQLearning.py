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

# Example usage

def calculatValueFunction(Q_table,e):

  v=np.zeros((5,5))
  for i in range (5):
    for j in range(5):
      s=convertCoordTostate(i,j)

      v[i][j]=np.max(Q_table[s])
  return v
def pi(state, a_,maxActionIndex,Q_table,e):
  s=0
  c=0
  max=np.max(Q_table[state])

  for ac in Q_table[state]:
    if ac==max:
      c=c+1

  if a_==Q_table[state][maxActionIndex]:
    if(c==4):
      s=(1-e)/c
    else:
      s=(1-e)/c + e/(4-c)
  else:
    s=e/(4-c)
  return s

most_optimal_value_function=[[4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
[4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
[3.8672, 4.3900, 0.0, 7.5769, 8.4637],
[3.4182, 3.8319, 0.0, 8.5738, 9.6946],
[2.9977, 2.9309, 6.0733, 9.6946, 0.0]]

print()
def mseFunction(value_function1, value_function2):
  error=0
  for i in range(5):
    for j in range(5):
      error=error + pow(value_function1[i][j]-value_function2[i][j],2)
  return error/25



def qlearning(s, a, r, s_):

    Q[s][a] =Q[s][a]+ alpha * (r + y * np.max(Q[s_]) - Q[s][a])

def convertCoordTostate(i,j):
  return i*5+j

def state_to_coordinates(state, grid_size=5):
    i = state // grid_size
    j = state % grid_size
    return i, j

def epsilon_greedy_policy(s1):
    i,j=state_to_coordinates(s1)
    if random.uniform(0, 1) < e or np.all(Q[s1] == 0):
        a=random.randint(0, 4 - 1)
        policy_table[i][j]=a
        return a
    else:
        policy_table[i][j]=np.argmax(Q[s1])
        return np.argmax(Q[s1])

def convertPolicyToTable(policy):

  for i in range(5):
    for j in range (5):
      if(policy[i][j]==0):
        policy[i][j]='U'
      elif (policy[i][j]==1):
        policy[i][j]='D'
      elif (policy[i][j]==2):
        policy[i][j]='L'

      elif(policy[i][j])==3:
        policy[i][j]='R'
  print(policy)
  direction_mapping = {'R': '→', 'D': '↓', 'U': '↑', 'L': '←', None: ' ','*':'G'}
  policy[4][4]='*'


  print()
  print()
  print("Optimal Policy")
  for row in policy:
      for direction in row:


        print(f"{direction_mapping[direction]:^4}", end="")
      print()




num_episodes = 2000

action=['U','D','L','R']
e=0.2
alpha=0.1
y=0.9
mse=[]
Q = np.zeros((25, 4))
policy_table=[[None for _ in range(5)] for _ in range(5)]
average_episode_counts = []
averageMse=[]
Q_list=[]
for i in range (0,20):

  total_steps = 0
  episode_counts = []
  Mse_error=[]
  for episode in range(num_episodes):
      s0 = d0()
      state0=convertCoordTostate(s0[0],s0[1])
      a = epsilon_greedy_policy(state0)

      steps_in_episode = 0
      while(True):

          next_state ,r = nextState(s0,action[a])

          state1=convertCoordTostate(next_state[0],next_state[1])

          next_action = epsilon_greedy_policy(state1)

          #next_action=action[next_action]
          state0=convertCoordTostate(s0[0],s0[1])

          qlearning(state0, a, r, state1)

          s0 = next_state
          a = next_action
          steps_in_episode += 1
          total_steps += 1
          if(s0[0]==4 and s0[1]==4):
            break;

      Q_list.append(Q)
      v=calculatValueFunction(Q,e)
     # print(v,most_optimal_value_function)
      mseError=mseFunction(v,most_optimal_value_function)
      Mse_error.append((mseError,episode+1))
      episode_counts.append((total_steps, episode + 1))
  steps,episodes=zip(*episode_counts)
  plt.plot(steps,episodes,'--',label=f"Run {i+1}")
  average_episode_counts.append(episode_counts)
  steps1,episodes1=zip(*Mse_error)
  averageMse.append(Mse_error)


average_curve = np.mean(average_episode_counts, axis=0)

print("Learned Q-values:")
print(Q)
policyTesting=policy_table
convertPolicyToTable(policy_table)
steps, episodes = zip(*average_curve)

plt.plot(steps, episodes,'-', label=' Average curve ',color='black')
plt.xlabel('Total Number of Steps')
plt.ylabel('Number of Episodes')
plt.legend()
plt.show()
average_mse = np.mean(averageMse, axis=0)
MSE_ERROR, episodes = zip(*average_mse)
plt.plot(episodes, MSE_ERROR,'-', label=' Average curve ',color='black')
plt.xlabel('Episodes')
plt.ylabel('MSEError')
plt.legend()
plt.show()
