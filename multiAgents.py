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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        #to calculate the final score
        score = 0
        #current gamestate data
        currentPos = successorGameState.getPacmanPosition()
        ghostStates = currentGameState.getGhostStates()
        currentFood = currentGameState.getFood()
        currentGhostStates = currentGameState.getGhostStates()
        ScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
        newFoodNum = len(newFood.asList())
        curFoodNum = len(currentFood.asList())

        #calculate pos to nearest food
        curFoodDist = []
        for food in currentFood.asList():
            curFoodDist.append(manhattanDistance(currentPos, food))
        #select the min
        minFoodDist = min(curFoodDist)

        #do the same for newPos
        newFoodDist = []
        for food in newFood.asList():
            newFoodDist.append(manhattanDistance(newPos, food))
        #select the min
        minNewFoodDist = min(curFoodDist)

        #if is gonna eat add the score
        if(newFoodNum < curFoodNum):
            score += 200
        #if you get closer to a food add score
        else:
            if(minNewFoodDist < minFoodDist):
                score += minNewFoodDist * 10
            else:
                score -= minNewFoodDist * 30

        #calculate current disctance from the nearest ghost
        curGhostDist = []
        for ghost in currentGhostStates:
            curGhostDist.append(manhattanDistance(currentPos, ghost.getPosition()))

        minDist = min(curGhostDist)

        #do it again for newpos
        newGhostDist = []
        for ghost in newGhostStates:
            newGhostDist.append(manhattanDistance(newPos, ghost.getPosition()))

        minNewDist = min(curGhostDist)

        #if the ghosts are scared in the current position
        #prefer the closest distance
        if(sum(ScaredTimes) > 0):
            if(minNewDist < minDist):
                score += (minDist - minNewDist) * 20
        #if you are going to get closer remove score
        else:
            #this is a lose state so return very low score
            if(minNewDist <= 1):
                score = -10000
            #if the new dist is very close to a ghost remove points
            elif(minNewDist <= 10):
                score -= minNewDist * (minDist - minNewDist)

        #penalty for stop
        if action == Directions.STOP:
            score -= 100

        #add score if you are gonna eat a capsule
        if(newPos in successorGameState.getCapsules()):
            score += 200

        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

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

        #we are following the lecture code
        #and adjusting it to the pacman game

        #we want the score and the cooresponding move
        #the function returns [score, move]
        def maxValue(agent, gameState, depth):
            #if it's a goal state return the evaluation score of the node
            if(depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin()):
                return [self.evaluationFunction(gameState)]

            #set variables for maxScore and maxMove
            maxScore = -float("inf")
            maxMove = "None"
            #get avaible actions
            actions = gameState.getLegalActions(agent)
            #for all the avaiable actions
            for action in actions:
                #get the successor
                succ = gameState.generateSuccessor(agent, action)
                #set the new agent
                #if we havent reached the max, go to next ghost
                #else to to pacman again
                if(agent == gameState.getNumAgents() - 1):
                    newAgent = 0
                else:
                    newAgent = agent + 1
                #call the master function
                score = minMaxDecision(newAgent, succ, depth + 1)[0]

                #keep the max score and move
                if(score >= maxScore):
                    maxScore = score
                    maxMove = action

            #return the max action along with the score
            return [maxScore, maxMove]

        
        def  minValue(agent, gameState, depth):
            #if it's a goal state return the evaluation score of the node
            if(depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin()):
                return [self.evaluationFunction(gameState)]
            #set variables for minScore and minMove
            minScore = float("inf")
            minMove = "None"
            #get avaible actions
            actions = gameState.getLegalActions(agent)
            #for all the avaiable actions
            for action in actions:
                #get the successor
                succ = gameState.generateSuccessor(agent, action)
                #set the new agent
                #if we havent reached the max, go to next ghost
                #else to to pacman again
                if(agent == gameState.getNumAgents() - 1):
                    newAgent = 0
                else:
                    newAgent = agent + 1
                #call the master function
                score = minMaxDecision(newAgent, succ, depth + 1)[0]

                #keep the max score and move
                if(score <= minScore):
                    minScore = score
                    minMove = action

            #return the max action along with the score
            return [minScore, minMove]

        def minMaxDecision(agent, gameState, depth):

            #if it's the pacman go for max
            #return [1] as the score is in pos [0]
            if(agent == 0):
                return maxValue(agent, gameState, depth)
            #if it's the ghost go for the min
            else:
                return minValue(agent, gameState, depth)

        #we want the best move for pacman so we must do 
        return maxValue(0, gameState, 0)[1]
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        #alpha-beta is extremly similiar with the min-max algorithm
        #we follow the a similar logic
        #we follow the lecture code

        def minValueAlphaBeta(agent, gameState, depth, a, b):
            #if it's a goal state return the evaluation score of the node
            if(depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin()):
                return [self.evaluationFunction(gameState)]

            #set variables for maxScore and maxMove
            minScore = float("inf")
            minMove = "None"
            #get avaible actions
            actions = gameState.getLegalActions(agent)
            #for all the avaiable actions
            for action in actions:
                #get the successor
                succ = gameState.generateSuccessor(agent, action)
                #set the new agent
                #if we havent reached the max, go to next ghost
                #else to to pacman again
                if(agent == gameState.getNumAgents() - 1):
                    newAgent = 0
                else:
                    newAgent = agent + 1
                #call the master function
                score = alphaBetaSearch(newAgent, succ, depth + 1, a, b)[0]

                if(score <= minScore):
                    minScore = score
                    minMove = action

                #all code above is the same as minMax

                #apply the alpha beta logic
                if(minScore < a):
                    return [minScore, minMove]

                b = min(b, minScore)
                
            #return the max action along with the score
            return [minScore, minMove]

        def maxValueAlphaBeta(agent, gameState, depth, a, b):
            #if it's a goal state return the evaluation score of the node
            if(depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin()):
                return [self.evaluationFunction(gameState)]

            #set variables for maxScore and maxMove
            maxScore = -float("inf")
            maxMove = "None"
            #get avaible actions
            actions = gameState.getLegalActions(agent)
            #for all the avaiable actions
            for action in actions:
                #get the successor
                succ = gameState.generateSuccessor(agent, action)
                #set the new agent
                #if we havent reached the max, go to next ghost
                #else to to pacman again
                if(agent == gameState.getNumAgents() - 1):
                    newAgent = 0
                else:
                    newAgent = agent + 1
                #call the master function
                score = alphaBetaSearch(newAgent, succ, depth + 1, a, b)[0]

                if(score >= maxScore):
                    maxScore = score
                    maxMove = action

                #all code above is the same as minMax

                #apply the alpha beta logic
                if(maxScore > b):
                    return [maxScore, maxMove]

                a = max(a, maxScore)

            #return the max action along with the score
            return [maxScore, maxMove]

        def alphaBetaSearch(agent, gameState, depth, a, b):

             #if it's the pacman go for max
            #return [1] as the score is in pos [0]
            if(agent == 0):
                return maxValueAlphaBeta(agent, gameState, depth, a, b)
            #if it's the ghost go for the min
            else:
                return minValueAlphaBeta(agent, gameState, depth, a, b)

        #we want the best move for pacman so we must do 
        #set alpha, beta
        alpha = -float("inf")
        beta = float("inf")
        return maxValueAlphaBeta(0, gameState, 0, alpha, beta)[1]


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

        #we are using the code from minMax question
        #but the ghosts chose randomly one of the moves
        #we need to change the minMax only for the ghost selection of score
        #for that part we follow the wiki code

        def maxValue(agent, gameState, depth):
            #if it's a goal state return the evaluation score of the node
            if(depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin()):
                return [self.evaluationFunction(gameState)]

            #set variables for maxScore and maxMove
            maxScore = -float("inf")
            maxMove = "None"
            #get avaible actions
            actions = gameState.getLegalActions(agent)
            #for all the avaiable actions
            for action in actions:
                #get the successor
                succ = gameState.generateSuccessor(agent, action)
                #set the new agent
                #if we havent reached the max, go to next ghost
                #else to to pacman again
                if(agent == gameState.getNumAgents() - 1):
                    newAgent = 0
                else:
                    newAgent = agent + 1
                #call the master function
                score = expectiMaxDecision(newAgent, succ, depth + 1)[0]

                #keep the max score and move
                if(score >= maxScore):
                    maxScore = score
                    maxMove = action

            #return the max action along with the score
            return [maxScore, maxMove]

        #the ghosts go for the random
        def  randomValue(agent, gameState, depth):
            #if it's a goal state return the evaluation score of the node
            if(depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin()):
                return [self.evaluationFunction(gameState)]
            #set variables for randomScore and move
            randScore = 0
            randMove = "None"
            #get avaible actions
            actions = gameState.getLegalActions(agent)

            #we want equal possibility for each action
            p = 100 / len(actions)

            #for all the avaiable actions
            for action in actions:
                #get the successor
                succ = gameState.generateSuccessor(agent, action)
                #set the new agent
                #if we havent reached the max, go to next ghost
                #else to to pacman again
                if(agent == gameState.getNumAgents() - 1):
                    newAgent = 0
                else:
                    newAgent = agent + 1
                #call the master function
                score = expectiMaxDecision(newAgent, succ, depth + 1)[0]

                #calculate the average score
                randScore += p * score

            #return the max action along with the score
            return [randScore]
            

        def expectiMaxDecision(agent, gameState, depth):

            #if it's the pacman go for max
            #return [1] as the score is in pos [0]
            if(agent == 0):
                return maxValue(agent, gameState, depth)
            #if it's the ghost go for the min
            else:
                return randomValue(agent, gameState, depth)

        #we want the best move for pacman so we must do 
        return maxValue(0, gameState, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

# Abbreviation
better = betterEvaluationFunction
