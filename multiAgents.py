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

        score = 0

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
        
        #following pseudocode found in the wiki
        def minMaxSearch(agent, depth, gameState):

            #if we have finished if its win or lose or we have reached the max depth
            if(gameState.isWin() == True) or (gameState.isLose() == True) or (depth == self.depth):
                #return the score of thr current state
                return [self.evaluationFunction(gameState)]

            #if we are at the last ghost change to pacman and add depth
            if(agent == gameState.getNumAgents() - 1):
                newAgent = 0
                depth += 1

            #if there are more ghosts go to the next one
            else:
                newAgent = agent + 1
            
            #if its the pacman agent
            #use the max score
            if(agent == 0):
                #to find the max score
                maxScore = -1000000
                maxMove = None
                #check the legal moves and find the best score, the best move
                for action in gameState.getLegalActions(agent):
                    #get the gamestate of the action
                    actionGameState = gameState.generateSuccessor(agent, action)
                    #recursive function call
                    #and get the score
                    #use [1] cause the output is (move, score) and we need only the score
                    score1 = minMaxSearch(newAgent, depth, actionGameState)[0]
                    #find the max
                    if(score1 >= maxScore):
                        maxScore = score1
                        #also add the maxMove
                        maxMove = action

                #return the two values found
                return [maxScore, maxMove]

            #if it's a ghost
            #use the min score
            else:
                minScore = 1000000
                minMove = None
                #check the legal moves and find the best score, the best move
                for action in gameState.getLegalActions(agent):
                    #get the gamestate of the action
                    actionGameState = gameState.generateSuccessor(agent, action)
                    #recursive function call
                    #and get the score
                    #use [1] cause the output is (move, score) and we need only the score
                    score2 = minMaxSearch(newAgent, depth, actionGameState)[0]
                    #find the min
                    if(score2 <= minScore):
                        minScore = score2
                        #also add the maxMove
                        move = action

                #return the two values found
                return [minScore, move]

        #use the function to find the best move
        #start with pacman so agent is 0
        #we starrt from depth 0
        #1 is the position of the move
        return minMaxSearch(0, 0, gameState)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
