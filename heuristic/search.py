# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def dfshelper(point, res, vis, problem):
    if point in vis:
        return False
    if problem.isGoalState(point):
        return True
    vis.add(point)
    for node in problem.getSuccessors(point):
        res.append(node[1])
        if dfshelper(node[0], res, vis, problem):
            return True
        res.pop()
    return False


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    from collections import deque
    result = deque()
    v = {None}
    st = problem.getStartState()
    ans = []
    if dfshelper(st, result, v, problem):
        for i in result:
            ans.append(i)
    return ans
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    result = []
    path = {}
    if problem.isGoalState(problem.getStartState()):
        return []

    queue = []
    queue.append(problem.getStartState())
    explored = {None}

    while queue:
        point = queue.pop(0)
        explored.add(point)
        if problem.isGoalState(point):
            node = point
            while node is not problem.getStartState():
                result.append(path[node][1])
                node = path[node][0]
            return result[::-1]
        for nex in problem.getSuccessors(point):
            if nex[0] not in explored and nex[0] not in queue:
                queue.append(nex[0])
                path[nex[0]] = [point, nex[1]]
    return []

    util.raiseNotDefined()


def getAct(n, problem, path):
    act = []
    while n is not problem.getStartState():
        act.append(path[n][1])
        n = path[n][0]
    return act[::-1]


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    path = {}
    if problem.isGoalState(problem.getStartState()):
        return []

    pq = util.PriorityQueue()
    pq.push(problem.getStartState(), problem.getCostOfActions([]))
    pqs = {None}
    pqs.add(problem.getStartState())
    explored = {None}

    while pq:
        point = pq.pop()
        pqs.remove(point)
        explored.add(point)
        if problem.isGoalState(point):
            node = point
            return getAct(node, problem, path)
        for nex in problem.getSuccessors(point):
            if nex[0] not in explored and nex[0] not in pqs:
                path[nex[0]] = [point, nex[1]]
                pq.push(nex[0], problem.getCostOfActions(getAct(nex[0], problem, path)))
                pqs.add(nex[0])
            if nex[0] in pqs:
                tmp=path[nex[0]]
                co=problem.getCostOfActions(getAct(nex[0], problem, path))
                path[nex[0]] = [point, nex[1]]
                cn=problem.getCostOfActions(getAct(nex[0], problem, path))
                if co<=cn:
                    path[nex[0]]=tmp
                pq.update(nex[0], problem.getCostOfActions(getAct(nex[0], problem, path)))
    return []

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    path = {}
    if problem.isGoalState(problem.getStartState()):
        return []

    pq = util.PriorityQueue()
    pq.push(problem.getStartState(), problem.getCostOfActions([])+heuristic(problem.getStartState(),problem))
    pqs = {None}
    pqs.add(problem.getStartState())
    explored = {None}

    while pq:
        point = pq.pop()
        pqs.remove(point)
        explored.add(point)
        if problem.isGoalState(point):
            node = point
            return getAct(node, problem, path)
        for nex in problem.getSuccessors(point):
            if nex[0] not in explored and nex[0] not in pqs:
                path[nex[0]] = [point, nex[1]]
                pq.push(nex[0], problem.getCostOfActions(getAct(nex[0], problem, path))+heuristic(nex[0],problem))
                pqs.add(nex[0])
            if nex[0] in pqs:
                tmp = path[nex[0]]
                co = problem.getCostOfActions(getAct(nex[0], problem, path))+heuristic(nex[0],problem)
                path[nex[0]] = [point, nex[1]]
                cn = problem.getCostOfActions(getAct(nex[0], problem, path))+heuristic(nex[0],problem)
                if co <= cn:
                    path[nex[0]] = tmp
                pq.update(nex[0], problem.getCostOfActions(getAct(nex[0], problem, path))+heuristic(nex[0],problem))
    return []
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
