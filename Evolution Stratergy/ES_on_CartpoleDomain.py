#Defining Constants
import numpy as np
import math
import random
import matplotlib.pyplot as plt
g = 9.8 #(gravity)
mc = 1.0 #(cart’s mass)
mp = 0.1 #(pole’s mass)
mt = mc + mp #(total mass)
l = 0.5 #(pole’s length)
tou =  0.02 #0.02
w=[]
w_dot=[]
#random.seed(42)
policy_time={}

def find_policy_cosine(theta, s, m):
  phi_s=[1]

  calculate_cosine_group(s[0], m,phi_s)
  calculate_cosine_group(s[1], m,phi_s)
  calculate_cosine_group(s[2], m,phi_s)
  calculate_cosine_group(s[3], m,phi_s)


  array1 = np.array(theta)
  array2 = np.array(phi_s)


  threshold= np.transpose(phi_s).dot(theta)

  if threshold > 0:
    return 10

  else:
    return -10

def calculate_cosine_group(value, m,phi_s):
  for k in range(1,m+1):
    phi_s.append(math.cos(k * math.pi * value))

def normaliZation(s):
  s_new=[0,0,0,0]

  s_new[0]=(s[0]+2.4)/4.8
  s_new[1]=(s[1]+math.pi/15)/(2*math.pi/15)
  s_new[2]=s[2]
  s_new[3]=s[3]
 
  return s_new

def calculate_sine_group(value, m,phi_s):
  for k in range(1,m+1):
    phi_s.append(math.sin(k * math.pi * value))

def SineNormaliZation(s):
  s_new=[0,0,0,0]
  s_new[0]=(s[0])/2.4
  s_new[1]=(s[1])/(math.pi/15)
  s_new[2]=s[2]
  s_new[3]=s[3]



  return s_new

def find_policy_sine(theta, s, m):
  phi_s=[1]

  calculate_sine_group(s[0], m,phi_s)
  calculate_sine_group(s[1], m,phi_s)
  calculate_sine_group(s[2], m,phi_s)
  calculate_sine_group(s[3], m,phi_s)


  array1 = np.array(theta)
  array2 = np.array(phi_s)


  threshold= np.transpose(phi_s).dot(theta)
  if threshold > 0:
    return 10

  else:
    return -10



def estimate_j(theta, N):
  G=[]
  for i in range(0,N):
    g=0
    s_0=[0,0,0,0]
    s_new=[0,0,0,0]
    for t in range (501):

       #x,v,w,w.

      normalized_state= SineNormaliZation(s_0)#normaliZation(s_0) # #
      F=find_policy_sine(theta,normalized_state,m)#find_policy_cosine(theta,normalized_state,m) #find_policy_sine(theta,normalized_state,m) #find_policy_cosine(theta,normalized_state,m)


      x=s_0[0]
      v=s_0[1]
      w=s_0[2]
      w_dot=s_0[3]



      # b= (-10 + mp * l* w_dot**2 * math.sin(w) ) /mt
      # c= (g*math.sin(w)- math.cos(w)*b)/(l*((4/3) - mp*math.cos(w)*math.cos(w)/mt))
      # d= b- (mp*l*c*math.cos(w)/mt)

      b = (F + mp * l *w_dot**2 * np.sin(w)) \
            / \
            mt

      c = (g * np.sin(w) - b * np.cos(w)) \
            / \
            (l * (4/3 - (mp * np.cos(w)**2)/mt))

      d = b - (mp * l * c * np.cos(w))/mt



      s_new[0]=x + v*tou

      s_new[1]=v+ tou*d

      s_new[2]=w+ tou*w_dot

      s_new[3]=w_dot+ tou*c

      for ind in range(0,4):
        s_0[ind]=s_new[ind]



      if(s_0[0] < -2.4 or s_0[0] > 2.4):
            break
      if(s_0[2] < -np.pi/15 or s_0[2] > np.pi/15):
            break
      g=g+1

    G.append(g)

  return sum(G)/N





m=25
n= 4*m+1
sigma=0.4 #0.2
alpha=0.009 #0.001
nPerturbation=200 #0
I= np.identity(n)
N=1
policy_iteration=300

avg_values = np.zeros(policy_iteration) 

for trials in range (0,5):
  theta = np.random.normal(0, 1, n)
  J_trials = np.zeros(policy_iteration)

  for t in range (0,policy_iteration):
    J={}
    e = np.zeros((nPerturbation, n))
    temp=[0]*n
    for i in range(0, nPerturbation):

      epsilon_i = np.random.multivariate_normal(np.zeros(n), I)
      e[i]=epsilon_i

      theta_new=theta + sigma*epsilon_i
      
      J[i]=estimate_j(theta_new, N)
      temp += list(J.values())[i] * e[i]
    
    avg_val = np.mean(list(J.values()))
    avg_values[t]=avg_values[t]+avg_val
    theta= theta + alpha*(1/(sigma*nPerturbation))*temp
    J_trials[t]=avg_val
     # Replace this with your actual J values
  t_values = range(policy_iteration) 
  
   # Assuming you have 500 iterations
  plt.plot(range(policy_iteration), J_trials, label=f'Trial {trials + 1}')
# Create a plot

for i in range(0,len(avg_values)):
  avg_values[i]=avg_values[i]/5

plt.plot(range(policy_iteration), avg_values, 'ro', label='Average', markersize=5)
  # Add labels and title
plt.xlabel('Iterations (t)')
plt.ylabel('J Values')
plt.title('J Values vs. Iterations')
plt.legend()
# Show the plot
plt.show()

