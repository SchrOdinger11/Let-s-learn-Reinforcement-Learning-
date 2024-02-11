import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
machine = "cuda" if torch.cuda.is_available() else "cpu"

#To test out Mountain Car Domain, change mdp=gym.make('CartPole-v1') in reinforceBaselineAlgorithm to 
#gym.make('MountainCar-v0')

class piNN(nn.Module):


    def __init__(self,  states,actions):

        super(piNN, self).__init__()
        self.firstLayer = nn.Linear(states, 256)
        self.FinalLayer = nn.Linear(256, actions)

    #forward pass
    def forward(self, state):

        state = self.firstLayer(torch.tensor(state))
        state = nn.functional.relu(state)
        policy = self.FinalLayer(state)
        ProbabilitiesPolicy = nn.functional.softmax(policy, dim=0)

        return ProbabilitiesPolicy

    def logValueAction(ProbabilitiesPolicy):
      return torch.log(ProbabilitiesPolicy)

class ValueFunctionNN(nn.Module):


    def __init__(self, observation_space):
        super(ValueFunctionNN, self).__init__()

        self.firstLayer = nn.Linear(observation_space, 256)
        self.FinalLayer = nn.Linear(256, 1)

    def forward(self, state):
        state = self.firstLayer(torch.tensor(state))
        state = nn.functional.relu(state)
        stateValues = self.FinalLayer(state)

        return stateValues

class RLAgent:
    def __init__(self, num_features,num_actions):
        self.policyNN = piNN( num_features,num_actions)
        self.valueNN = ValueFunctionNN(num_features)
        # self.learning_rate_policy = 0.9
        # self.learning_rate_value=0.3
        self.policy_optimizer = torch.optim.Adam(self.policyNN.parameters(), lr=1e-2)#
        self.stateval_optimizer = torch.optim.Adam(self.valueNN.parameters(), lr=1e-2)#

    def nextAction(self, state):  #state is tensor
        nextActionProbabilities = self.policyNN(state)

        # if np.isnan(nextActionProbabilities).any():

        #   nextActionProbabilities = np.ones(len(nextActionProbabilities)) / len(nextActionProbabilities)
        nextActionProbabilitiesnumpy=nextActionProbabilities.detach().numpy()

        action =np.random.choice(len(nextActionProbabilitiesnumpy), p=nextActionProbabilitiesnumpy)
        return action,nextActionProbabilities

    def TotalReturns(self, rewards, gamma=0.99):
        G = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            G[t] = running_add
        return G

    def trainValueWeights(self,G, optimizer,stateValues ):


      #calculate MSE loss

      trainingLoss = nn.functional.mse_loss(stateValues, torch.tensor(G))

      #Backpropagate
      optimizer.zero_grad()
      trainingLoss.backward()
      optimizer.step()

    def trainPolicyWeights(self,deltas, optimizer, logProbabilities):



      policy_loss = []

     #grad descent
      for dell, log_value_action in zip(deltas, logProbabilities):

          policy_loss.append(-dell * log_value_action)

      #Backpropagation
      optimizer.zero_grad()
      sum(policy_loss).backward()
      optimizer.step()

def reinforceBaselineAlgorithm(num_episodes=1000,gamma=0.99):
    
    #gym.make('MountainCar-v0')
    mdp = gym.make('CartPole-v1')
    totalMDPActions = mdp.action_space.n
    totalMDPStates = mdp.observation_space.shape[0]




    agent = RLAgent(totalMDPStates,totalMDPActions)

    EpisodicRewards=[]

    for episode in range(num_episodes):
        simulated_states=[]
        simulated_actions=[]
        simulated_rewards =[]
        log_actions=[]

        s = mdp.reset()

        while True:

            action, probAction = agent.nextAction(s)

            s_next, r, EpisodeTerminate, info = mdp.step(action)

            simulated_states.append(s)
            simulated_actions.append(action)
            simulated_rewards.append(r)
            log_actions.append(probAction[action])



            if EpisodeTerminate:

                stateTensors=[]
                for states in simulated_states:
                  states = torch.from_numpy(states).float().unsqueeze(0).to(machine)

                  stateTensors.append(agent.valueNN(states))
                stateTensors = torch.stack(stateTensors).squeeze()


                agent.trainValueWeights(agent.TotalReturns(simulated_rewards,gamma),  agent.stateval_optimizer,stateTensors)
                advantage = [G_t - B for G_t, B in zip(agent.TotalReturns(simulated_rewards), stateTensors)]
                advantage = torch.tensor(advantage).to(machine)

                agent.trainPolicyWeights(advantage, agent.policy_optimizer, log_actions)
                #agent.update_weights(simulated_states, simulated_actions, simulated_rewards,probAction)
                reward_thisEpisode = sum(simulated_rewards)
                EpisodicRewards.append(reward_thisEpisode)

                break

            s = s_next

    mdp.close()



    plt.plot(EpisodicRewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward vs Episode - REINFORCE with Baseline')
    plt.show()

if __name__ == "__main__":
    reinforceBaselineAlgorithm()

