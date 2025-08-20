# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.Qvalues = util.Counter() # A Counter is a dict with default 0
        

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        for q in self.Qvalues:
          state_q = q[0]
          if state_q == state:
              return self.Qvalues[(state, action)]
          
        return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #check if we are at a terminal state 
        if state == 'TERMINAL_STATE':
          return 0.0 
        
        #if it is not terminal state 
    
        actions = self.getLegalActions(state)

        if not actions:
            return 0.0
        
        
        #iterate over actions and calculate the q-value 
        largest= max([self.getQValue(state, action) for action in actions])
        
        return largest
        
            
        

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        if state == 'TERMINAL_STATE':
            return None 
        
        actions =self.getLegalActions(state)
        action_dict = []
        best_val = self.computeValueFromQValues(state)
        
        for action in actions: 
            if self.getQValue(state, action) == best_val:
                action_dict.append(action)
        
        if action_dict:
            choice = random.choice(action_dict)
            #print(choice)
            return choice
        
        return None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        probability = self.epsilon
        random_choice = util.flipCoin(probability)

        #check if we're in a terminal state
        if state == 'TERMINAL_STATE':
            return None 
        
        if random_choice:
            actions = self.getLegalActions(state)
            action = random.choice(actions)
            
        
        else: 
            action = self.getPolicy(state)
        
        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        #we are updating self.qvalues
        #we are doing q-value update here 
        #argmax = T(s,a, s'){R(s,a,s') + gamma*V(S')}
        #
        state_reward = reward
        gamma = self.discount
        alpha = self.alpha
        #V(S')
        Q_max = self.getValue(nextState)
        Q_value =self.Qvalues[(state, action)]
        sample = reward + gamma * (Q_max)
      
        #now updating our self.Qvalues
        self.Qvalues [(state, action)] = (1-alpha) * Q_value + alpha * sample 

        
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #make array of weights 
        
        #make array of features 
        weights = np.array(self.getWeights())
        extractor = self.featExtractor
        features = np.array(extractor.getFeatures(state, action))
        
        return np.dot(weights, features)

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        state_reward = reward
        gamma = self.discount
        alpha = self.alpha
        #V(S')
        Q_max = self.getValue(nextState)
        Q_value =self.getQValue(state, action)
        difference = (reward + gamma * Q_max) - Q_value 
        extractor = self.featExtractor
        features = extractor.getFeatures(state, action)

        for feature in features:
            self.weights[feature] += alpha * difference * features[feature]
        

       
      
        #now updating our self.Qvalues


    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass