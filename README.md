### README
---
**Reinforcement Learning with the Lunar Lander**

### Intro

OpenAI Gym provides a number of environments for experimenting and testing reinforcement learning algorithms. 

Here I wanted to explore implementing a Double Deep Q Learning Network (DDQN) and a Deep Deterministic Policy Gradient (DDPG) on the discrete and continuous lunar lander environments.

The environments can be found: https://gym.openai.com/envs/#box2d

The environments can be installed via pip:

```pip install gym```


### Set up

Both the DDQN and DDPG are implemented in Jupyter Notebooks to allow for easy exploration and experiementation.


### Double Deep Q Learning Network (DDQN)

Deep Q Networks predict the Q value of each possible action. The agent will then pick the action with the largest Q value. With a basic Q learner,the Q value for each action at each state is calculated. By using a neural network with the current state as input a much larger state space can be managed. 

Deep Q learning relies on the Bellman equation to provide the loss.

The Bellman equation is 

Q(s,a) = R(s,a) + gamma*maxQ(next_state,a)


To train a DQN create a buffer of transitions. Transitions are sets of [state, action, next_state, reward]

To allow for exploration the action chosen is sometimes random, the rest of the time the action is chosen by the agent .The agent is the DQN and the action would simply be the one that leads to the biggest Q value. 

With each transition you have all the values in the bellman equation. The Q(s,a) is taken from the DQN using the state and action from the transition. R(s,a) is the reward. maxQ(next_state,a) is then simply using the DQN on the next state, what is the largest Q value of the possible actions. 

The loss is then simply the difference between the two sides of the bellman equation. Essentially how well is the DQN satisfying the bellman equation.

loss = Q(s,a) - (R(s,a) + gamma*maxQ(next_state,a))

The cost used is then the mean squared error of these loss terms for the batch.

The DQN is then trained on this cost. 

This implementation is a Double Deep Q Network. What this means is that a separate instance of the DQN is used to calculate the Q values of the next state to the one used to calculate the Q values of the current state. The network is the same but only one of them is trained in the process. The other network periodically inherits the trained weights of the other network. This offset helps the networks to train. 

A deep Q network can only be used for discrete action spaces as the network has to create a Q value for all actions at the current state. 

A DQN is off policy in that the replay buffer will contain example transitions created from a policy that is no longer optimum.

Using a replay buffer allow the network to not overly focus on what it has just seen and forget about useful experiences it has seen in the past. 


### Deep Deterministic Policy Gradient (DDPG)

A deep deterministic policy gradient uses a network to predict the action that should be taken at each step. By directly predicting the action it can handle much larger state spaces and continuous action profiles. 

A deep deterministic policy gradient is a form of actor critic. In fact there are two actors and two critic networks.

The actor networks predict the action to be taken for each state. The critic networks predict the Q value of taking a specific action at a given state. 

You have a critic network and a target critic network. This part is similar to the DQN and you train the critic network using the bellman equation. 

The intention of all reinforcement learning is to optimise the reward or the Q value. The critic networks predict the Q value based on the actions taken by the actor networks. To train the actor networks therefore, instead of having a traditional 'loss' you use the critic. By completing gradient ascent of the critic with respect the the actor you can optimise the actor.

In a similar fashion to the DQN, there are two critic and two actor networks. Poliac averaging is used to update the support networks.

To add exploration you add some noise to the action selection. 


### References

The DDQN is based on the following pytorch implementation : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


