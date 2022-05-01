# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        sc = 0
        d = 2 ** 31
        j = 0
        for i in newGhostStates:
            if newScaredTimes[j] == 0:
                d = min(d, util.manhattanDistance(newPos, i.getPosition()))
            j += 1
        dfood = 2 ** 31
        for f in currentGameState.getFood().asList():
            dfood = min(dfood, util.manhattanDistance(newPos, f))
        if d == 0:
            sc -= 1000
        else:
            sc -= 500 / d

        if dfood == 0:
            sc += 300
        else:
            sc += 100 / dfood

        return sc


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


def minimax(state, turn, depth, agent):
    best = None
    if turn == 0:
        bestscore = -2 ** 31
    else:
        bestscore = 2 ** 31

    if depth == 0 or state.isWin() or state.isLose():
        return [None, agent.evaluationFunction(state)]
    for act in state.getLegalActions(turn):
        s = state.generateSuccessor(turn, act)
        if turn == 0:
            score = minimax(s, 1, depth, agent)
        if turn != 0:
            if turn == state.getNumAgents() - 1:
                score = minimax(s, 0, depth - 1, agent)
            else:
                score = minimax(s, turn + 1, depth, agent)

        if turn == 0:

            if score[1] > bestscore:
                bestscore = score[1]
                best = act
        else:
            if score[1] < bestscore:
                bestscore = score[1]
                best = act

    return [best, bestscore]


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        res = minimax(gameState, 0, self.depth, self)
        return res[0]

        util.raiseNotDefined()


def alphabeta(state, turn, depth, alpha, beta, agent):
    best = None
    if turn == 0:
        bestscore = -2 ** 31
    else:
        bestscore = 2 ** 31

    if depth == 0 or state.isWin() or state.isLose():
        return [None, agent.evaluationFunction(state)]
    for act in state.getLegalActions(turn):
        s = state.generateSuccessor(turn, act)
        if turn == 0:
            score = alphabeta(s, 1, depth, alpha, beta, agent)
        if turn != 0:
            if turn == state.getNumAgents() - 1:
                score = alphabeta(s, 0, depth - 1, alpha, beta, agent)
            else:
                score = alphabeta(s, turn + 1, depth, alpha, beta, agent)

        if turn == 0:

            if score[1] > bestscore:
                bestscore = score[1]
                best = act
                if score[1] > beta:
                    return [best, bestscore]
                alpha = max(alpha, bestscore)
        else:
            if score[1] < bestscore:
                bestscore = score[1]
                best = act
                if score[1] < alpha:
                    return [best, bestscore]
                beta = min(beta, bestscore)

    return [best, bestscore]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        res = alphabeta(gameState, 0, self.depth, -2 ** 31, 2 ** 31, self)
        return res[0]
        util.raiseNotDefined()


def expect(state, turn, depth, agent):
    best = None
    if turn == 0:
        bestscore = -2 ** 31
    else:
        ghostsc = 0
    if depth == 0 or state.isWin() or state.isLose():
        return [None, agent.evaluationFunction(state)]
    for act in state.getLegalActions(turn):
        s = state.generateSuccessor(turn, act)
        if turn == 0:
            score = expect(s, 1, depth, agent)
        if turn != 0:
            if turn == state.getNumAgents() - 1:
                score = expect(s, 0, depth - 1, agent)
            else:
                score = expect(s, turn + 1, depth, agent)

        if turn == 0:

            if score[1] > bestscore:
                bestscore = score[1]
                best = act
        else:
            ghostsc += score[1]
    if turn != 0:
        l = state.getLegalActions(turn)
        avg = ghostsc / len(l)
        import random
        idx = random.randint(0, len(l) - 1)
        return [l[idx], avg]

    return [best, bestscore]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        res = expect(gameState, 0, self.depth, self)
        return res[0]
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Get the distance to the closest normal ghost. Score will decrease dramatically if the distance is very short.
    Being caught by the ghost loses 1000 points.
    Get the distance to the closest scared ghost. Score will increase if the pacman gets closer to the scared ghost
    but maintains a safe distance of 3.
    Get the distance to the closest food. Score will increase if the food is very close.
    Get the number of food. Score will have less deficit when there is less food remained.
    Get the distance to the closest capsule. Score will increase if the pacman can get a capsule when a ghost approaches.
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    sc = 0
    d = 2 ** 31
    d2 = 2 ** 31
    j = 0
    for i in newGhostStates:
        if newScaredTimes[j] == 0:
            d = min(d, util.manhattanDistance(newPos, i.getPosition()))
        else:
            d2 = min(d2, util.manhattanDistance(newPos, i.getPosition()))
        j += 1
    if d == 0:
        sc -= 1000
    else:
        sc -= 500 / d
    if d2 > 3:
        sc += 100 / d2

    dfood = 2 ** 31
    for f in currentGameState.getFood().asList():
        dfood = min(dfood, util.manhattanDistance(newPos, f))
    sc += 100 / dfood
    sc -= 400 * currentGameState.getNumFood()

    dcap = 2 ** 31
    for c in currentGameState.getCapsules():
        dcap = min(dcap, util.manhattanDistance(c, newPos))
    if d < 3:
        sc += 100 / dcap

    return sc
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
