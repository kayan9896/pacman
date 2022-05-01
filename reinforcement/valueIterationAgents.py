# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        cp = util.Counter()

        for i in range(0, self.iterations):
            for i in self.values:
                cp[i] = self.values[i]
            for s in self.mdp.getStates():
                if self.mdp.isTerminal(s):
                    continue
                qm = -2 ** 31
                for a in self.mdp.getPossibleActions(s):
                    q = 0
                    for ns, pb in self.mdp.getTransitionStatesAndProbs(s, a):
                        q += pb * (self.mdp.getReward(s, a, ns) + self.discount * cp[ns])
                    if q > qm:
                        qm = q

                self.values[s] = qm

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0
        for ns, pb in self.mdp.getTransitionStatesAndProbs(state, action):
            q += pb * (self.mdp.getReward(state, action, ns) + self.discount * self.values[ns])
        return q
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        qm = -2 ** 31
        move = None
        for a in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, a)
            if q > qm:
                qm = q
                move = a
        return move
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        for i in range(0, self.iterations):

            s = self.mdp.getStates()[i % len(self.mdp.getStates())]
            if self.mdp.isTerminal(s):
                continue
            qm = -2 ** 31
            for a in self.mdp.getPossibleActions(s):
                q = 0
                for ns, pb in self.mdp.getTransitionStatesAndProbs(s, a):
                    q += pb * (self.mdp.getReward(s, a, ns) + self.discount * self.values[ns])
                if q > qm:
                    qm = q

            self.values[s] = qm


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pre = dict()

        pq = util.PriorityQueue()
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s): continue
            qm = -2 ** 31
            for a in self.mdp.getPossibleActions(s):
                q = 0
                for ns, pb in self.mdp.getTransitionStatesAndProbs(s, a):
                    q += pb * (self.mdp.getReward(s, a, ns) + self.discount * self.values[ns])
                    if not self.mdp.isTerminal(ns):
                        tmp = pre.get(ns, [])
                        tmp.append(s)
                        pre[ns] = tmp
                qm = max(qm, q)

            diff = abs(self.values[s] - qm)
            if diff > self.theta:
                pq.push(s, -diff)

        for i in range(0, self.iterations):
            if not pq.isEmpty():
                s = pq.pop()
                qm1 = -2 ** 31
                for a in self.mdp.getPossibleActions(s):
                    q1 = self.computeQValueFromValues(s, a)
                    qm1 = max(q1, qm1)
                self.values[s] = qm1

                for p in pre[s]:
                    qm2 = -2 ** 31
                    for a in self.mdp.getPossibleActions(p):
                        q2 = self.computeQValueFromValues(p, a)
                        qm2 = max(qm2, q2)
                    diff = abs(self.values[p] - qm2)
                    if diff > self.theta:
                        pq.update(p, -diff)
