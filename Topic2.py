from math import sqrt
import numpy as np
import pandas as pd
from pygame.sprite import collide_circle_ratio
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from csv import reader
from math import sqrt
import random
import pygame
import random
import sys
import math


# region SearchAlgorithms
class Node:
    id = None  # Unique value for each node.
    up = None  # Represents value of neighbors (up, down, left, right).
    down = None
    left = None
    right = None
    previousNode = None  # Represents value of neighbors.

    def __init__(self, value):
        self.value = value


class SearchAlgorithms:
    ''' * DON'T change Class, Function or Parameters Names and Order
        * You can add ANY extra functions,
          classes you need as long as the main
          structure is left as is '''
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    mazeStr = None
    rowNum, colNum = 0, 0
    nodes = [[]]
    Visited = []
    Not_Visited = []
    E_id = None
    def __init__(self, mazeStr):
        ''' mazeStr contains the full board
         The board is read row wise,
        the nodes are numbered 0-based starting
        the leftmost node'''
        self.mazeStr = mazeStr
        self.ArrayInitialzer()

    def DFS(self):
        # Fill the correct path in self.path
        # self.fullPath should contain the order of visited nodes
        # self.path should contain the direct path from start node to goal node
        self.DFS_Helper(self.Not_Visited.pop())
        self.GetPath(self.nodes[self.E_id // self.colNum][self.E_id % self.colNum])
        return self.fullPath, self.path

    def DFS_Helper(self,node):

        self.fullPath.append(node.id)
        row_index , col_index = node.id // self.colNum , node.id % self.colNum
        if node.value == 'E':
             return True
        if node.up:
           if self.nodes[row_index - 1][col_index].value != '#' and (
                   self.nodes[row_index - 1][col_index].id) not in self.fullPath:
              flag = True
              # for i in  self.Not_Visited:
              #      if i.id == self.nodes[row_index - 1][col_index].id:
              #            flag = Fals
              if flag:
                  self.nodes[row_index - 1][col_index].previousNode = node
                  self.FindMov(row_index - 1,col_index,self.rowNum,self.colNum,self.nodes[row_index - 1][col_index])
                  self.DFS_Helper(self.nodes[row_index - 1][col_index])

        if node.down:
            if self.nodes[row_index + 1][col_index].value != '#' and (
                    self.nodes[row_index + 1][col_index].id) not in self.fullPath:
                flag = True
                # for i in self.Not_Visited:
                #     if i.id == self.nodes[row_index + 1][col_index].id:
                #         flag = False
                if flag:
                    self.nodes[row_index + 1][col_index].previousNode = node
                    self.FindMov(row_index + 1, col_index, self.rowNum, self.colNum,
                                 self.nodes[row_index + 1][col_index])
        #            self.Not_Visited.insert(0, self.nodes[row_index + 1][col_index])
                    self.DFS_Helper(self.nodes[row_index + 1][col_index])
        if node.left:
            if self.nodes[row_index][col_index - 1].value != '#' and (
                    self.nodes[row_index][col_index - 1].id) not in self.fullPath:
                flag = True
                # for i in  self.Not_Visited:
                #     if i.id == self.nodes[row_index][col_index - 1].id:
                #         flag = False
                if flag:
                    self.nodes[row_index][col_index - 1].previousNode = node
                    self.FindMov(row_index, col_index - 1, self.rowNum, self.colNum,
                                 self.nodes[row_index][col_index - 1])
                    self.DFS_Helper(self.nodes[row_index][col_index - 1])
        if node.right:
            if self.nodes[row_index][col_index + 1].value != '#' and (
                    self.nodes[row_index][col_index + 1].id) not in self.fullPath:
                flag = True
                # for i in self.Not_Visited:
                #     if i.id == self.nodes[row_index][col_index + 1].id:
                #         flag = False
                if flag:
                    self.nodes[row_index][col_index + 1].previousNode = node
                    self.FindMov(row_index, col_index + 1, self.rowNum, self.colNum,
                                 self.nodes[row_index][col_index + 1])
                    self.DFS_Helper(self.nodes[row_index][col_index + 1])

    def GetPath(self,node):
        while node.previousNode != None:
            self.path.append(node.id)
            node = node.previousNode
        self.path.append(node.id)
        self.path.reverse()

    def ArrayInitialzer(self):
        str = self.mazeStr.split()
        self.rowNum = len(str)
        self.colNum = len(str[0])
        self.colNum //= 2
        self.colNum += 1
        self.nodes = [[None]*self.colNum for _ in range(self.rowNum)]

        count = 0
        for i in range(self.rowNum):
            colIndex = 0
            for j in range(len(str[0])):
                if str[i][j] != ',':
                    self.nodes[i][colIndex] = Node(str[i][j])
                    self.nodes[i][colIndex].id = count
                    if str[i][j] == 'S':
                        self.FindMov(i, j, self.rowNum, self.colNum, self.nodes[i][colIndex])
                        self.Not_Visited.append(self.nodes[i][colIndex])
                    if str[i][j] == 'E':
                        self.E_id = count
                    count += 1
                    colIndex += 1

    def FindMov(self,rowIndex,colIndex,RowSize,ColSize,Node):
        if rowIndex - 1 >= 0:
            Node.up = True
        if rowIndex + 1 < RowSize:
            Node.down = True
        if colIndex - 1 >= 0:
            Node.left = True
        if colIndex + 1 < ColSize:
            Node.right = True

# endregion

#region Gaming
class Gaming:
    def __init__(self):
        self.COLOR_BLUE = (0, 0, 240)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_RED = (255, 0, 0)
        self.COLOR_YELLOW = (255, 255, 0)

        self.Y_COUNT = int(5)
        self.X_COUNT = int(8)

        self.PLAYER = 0
        self.AI = 1

        self.PLAYER_PIECE = 1
        self.AI_PIECE = 2

        self.WINNING_WINDOW_LENGTH = 3
        self.EMPTY = 0
        self.WINNING_POSITION = []
        self.SQUARESIZE = 80

        self.width = self.X_COUNT * self.SQUARESIZE
        self.height = (self.Y_COUNT + 1) * self.SQUARESIZE

        self.size = (self.width, self.height)

        self.RADIUS = int(self.SQUARESIZE / 2 - 5)

        self.screen = pygame.display.set_mode(self.size)


    def create_board(self):
        board = np.zeros((self.Y_COUNT, self.X_COUNT))
        return board


    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece


    def is_valid_location(self, board, col):
        return board[self.Y_COUNT - 1][col] == 0


    def get_next_open_row(self, board, col):
        for r in range(self.Y_COUNT):
            if board[r][col] == 0:
                return r


    def print_board(self, board):
        print(np.flip(board, 0))


    def winning_move(self, board, piece):
        self.WINNING_POSITION.clear()
        for c in range(self.X_COUNT - 2):
            for r in range(self.Y_COUNT):
                if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r, c + 1])
                    self.WINNING_POSITION.append([r, c + 2])
                    return True

        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT - 2):
                if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r + 1, c])
                    self.WINNING_POSITION.append([r + 2, c])
                    return True

        for c in range(self.X_COUNT - 2):
            for r in range(self.Y_COUNT - 2):
                if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r + 1, c + 1])
                    self.WINNING_POSITION.append([r + 2, c + 2])
                    return True

        for c in range(self.X_COUNT - 2):
            for r in range(2, self.Y_COUNT):
                if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r - 1, c + 1])
                    self.WINNING_POSITION.append([r - 2, c + 2])
                    return True


    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = self.PLAYER_PIECE
        if piece == self.PLAYER_PIECE:
            opp_piece = self.AI_PIECE

        if window.count(piece) == 3:
            score += 100
        elif window.count(piece) == 2 and window.count(self.EMPTY) == 1:
            score += 5

        if window.count(opp_piece) == 3 and window.count(self.EMPTY) == 1:
            score -= 4

        return score


    def score_position(self, board, piece):
        score = 0
        center_array = [int(i) for i in list(board[:, self.X_COUNT // 2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        for r in range(self.Y_COUNT):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(self.X_COUNT - 3):
                window = row_array[c: c + self.WINNING_WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        for c in range(self.X_COUNT):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(self.Y_COUNT - 3):
                window = col_array[r: r + self.WINNING_WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        for r in range(self.Y_COUNT - 3):
            for c in range(self.X_COUNT - 3):
                window = [board[r + i][c + i] for i in range(self.WINNING_WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        for r in range(self.Y_COUNT - 3):
            for c in range(self.X_COUNT - 3):
                window = [board[r + 3 - i][c + i] for i in range(self.WINNING_WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score


    def is_terminal_node(self, board):
        return self.winning_move(board, self.PLAYER_PIECE) or self.winning_move(board, self.AI_PIECE) or len(
                self.get_valid_locations(board)) == 0



    def AlphaBeta(self, board, depth, alpha, beta, currentPlayer):
        valid_locations = self.get_valid_locations(board)
        value = -math.inf
        column = random.choice(valid_locations)
        '''Implement here'''

        if self.winning_move(board,self.AI_PIECE):
          return None, 1000000   #self.score_position(board, self.AI_PIECE)
        elif self.winning_move(board,self.PLAYER_PIECE):
          return None, -1 * 1000000  #self.score_position(board,self.PLAYER_PIECE)
        elif len(self.get_valid_locations(board)) == 0:
          return None, 0

        if depth == 0:
            if currentPlayer:
                return None, self.score_position(board, self.AI_PIECE)
            else:
                return None, self.score_position(board, self.PLAYER_PIECE)

        if currentPlayer:
            for col in valid_locations:
                #CopyFBoard = board.copy()
                row = self.get_next_open_row(board, col)
                self.drop_piece(board, row, col, self.AI_PIECE)
                ReturnedColumn, currentValue = self.AlphaBeta(board, depth - 1, alpha, beta, not currentPlayer)
                #value = max(value, currentValue)
                self.drop_piece(board, row, col, self.EMPTY)
                if currentValue > value:
                    value = currentValue
                    column = col

                alpha = max(alpha, value)
                if beta <= alpha:
                    break

        else:
            value = math.inf
            for col in valid_locations:
                #CopyFBoard = board.copy()
                row = self.get_next_open_row(board, col)
                self.drop_piece(board, row, col, self.PLAYER_PIECE)
                ReturnedColumn, currentValue = self.AlphaBeta(board, depth - 1, alpha, beta, not currentPlayer)
                #value = min(value, currentValue)
                self.drop_piece(board, row, col, self.EMPTY)
                beta = min(beta, value)
                if currentValue < value:
                    value = currentValue
                    column = col
                if beta <= alpha:
                    break

        return column, value

    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(self.X_COUNT):
            if self.is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations


    def pick_best_move(self, board, piece):
        best_score = -10000
        valid_locations = self.get_valid_locations(board)
        best_col = random.choice(valid_locations)

        for col in valid_locations:
            row = self.get_next_open_row(board, col)
            temp_board = board.copy()
            self.drop_piece(temp_board, row, col, piece)
            score = self.score_position(temp_board, piece)

            if score > best_score:
                best_score = score
                best_col = col

        return best_col


    def draw_board(self, board):
        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT):
                pygame.draw.rect(self.screen, self.COLOR_BLUE,
                                 (c * self.SQUARESIZE, r * self.SQUARESIZE + self.SQUARESIZE, self.SQUARESIZE,
                                  self.SQUARESIZE))
                pygame.draw.circle(self.screen, self.COLOR_BLACK, (
                        int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                        int(r * self.SQUARESIZE + self.SQUARESIZE + self.SQUARESIZE / 2)),
                                   self.RADIUS)

        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT):
                if board[r][c] == self.PLAYER_PIECE:
                    pygame.draw.circle(self.screen, self.COLOR_RED, (
                            int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                            self.height - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)),
                                       self.RADIUS)
                elif board[r][c] == self.AI_PIECE:
                    pygame.draw.circle(self.screen, self.COLOR_YELLOW, (
                            int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                            self.height - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)),
                                       self.RADIUS)
        pygame.display.update()
#endregion

# region KMEANS
class DataItem:
    def __init__(self, item):
        self.features = item
        self.clusterId = -1

    def getDataset():
        data = []
        data.append(DataItem([0, 0, 0, 0]))
        data.append(DataItem([0, 0, 0, 1]))
        data.append(DataItem([0, 0, 1, 0]))
        data.append(DataItem([0, 0, 1, 1]))
        data.append(DataItem([0, 1, 0, 0]))
        data.append(DataItem([0, 1, 0, 1]))
        data.append(DataItem([0, 1, 1, 0]))
        data.append(DataItem([0, 1, 1, 1]))
        data.append(DataItem([1, 0, 0, 0]))
        data.append(DataItem([1, 0, 0, 1]))
        data.append(DataItem([1, 0, 1, 0]))
        data.append(DataItem([1, 0, 1, 1]))
        data.append(DataItem([1, 1, 0, 0]))
        data.append(DataItem([1, 1, 0, 1]))
        data.append(DataItem([1, 1, 1, 0]))
        data.append(DataItem([1, 1, 1, 1]))
        return data

class Cluster:
        def __init__(self, id, centroid):
            self.centroid = centroid
            self.data = []
            self.id = id

        def update(self, clusterData):
            self.data = []
            for item in clusterData:
                self.data.append(item.features)
            tmpC = np.average(self.data, axis=0)
            tmpL = []
            for i in tmpC:
                tmpL.append(i)
            self.centroid = tmpL

class SimilarityDistance:
        def euclidean_distance(self, p1, p2):
            distance = 0
            for i in range(len(p1)):
                distance += (p1[i] - p2[i]) ** 2
            finalResult = sqrt(distance)
            return finalResult

        def Manhattan_distance(self, p1, p2):
            distance = 0
            for i in range(len(p1)):
               distance += abs(p1[i] - p2[i])
            return distance

class Clustering_kmeans:
        def __init__(self, data, k, noOfIterations, isEuclidean):
            self.data = data
            self.k = k
            self.distance = SimilarityDistance()
            self.noOfIterations = noOfIterations
            self.isEuclidean = isEuclidean

        def initClusters(self):
            self.clusters = []
            for i in range(self.k):
                self.clusters.append(Cluster(i, self.data[i * 10].features))

        def getClusters(self):
            self.initClusters()
            '''Implement Here'''
            self.initClusters()
            for i in range(self.noOfIterations):
                for item in self.data:
                    itemDistance = math.inf
                    for c in self.clusters:
                        if self.isEuclidean:
                            ClusterDistance = self.distance.euclidean_distance(c.centroid, item.features)
                        elif not self.isEuclidean:
                            ClusterDistance = self.distance.Manhattan_distance(c.centroid, item.features)
                        if ClusterDistance < itemDistance:
                            itemDistance = ClusterDistance
                            item.clusterId = c.id
                    clusterItems = [itm for itm in self.data if itm.clusterId == item.clusterId]
                    self.clusters[item.clusterId].update(clusterItems)
            return self.clusters

# endregion

#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn
def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    fullPath, path = searchAlgo.DFS()
    print('**DFS**\n Full Path is: ' + str(fullPath) +'\n Path is: ' + str(path))

# endregion

#region Gaming_Main_fn
def Gaming_Main():
    game = Gaming()
    board = game.create_board()
    game.print_board(board)
    game_over = False

    pygame.init()

    game.draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 50)

    turn = 1

    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(game.screen, game.COLOR_BLACK, (0, 0, game.width, game.SQUARESIZE))
                posx = event.pos[0]
                if turn == game.PLAYER:
                    pygame.draw.circle(game.screen, game.COLOR_RED, (posx, int(game.SQUARESIZE / 2)), game.RADIUS)

            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(game.screen, game.COLOR_BLACK, (0, 0, game.width, game.SQUARESIZE))

                if turn == game.PLAYER:
                    posx = event.pos[0]
                    col = int(math.floor(posx / game.SQUARESIZE))

                    if game.is_valid_location(board, col):
                        row = game.get_next_open_row(board, col)
                        game.drop_piece(board, row, col, game.PLAYER_PIECE)

                        if game.winning_move(board, game.PLAYER_PIECE):
                            label = myfont.render("Player Human wins!", 1, game.COLOR_RED)
                            print(game.WINNING_POSITION)
                            game.screen.blit(label, (40, 10))
                            game_over = True

                        turn += 1
                        turn = turn % 2

                        # game.print_board(board)
                        game.draw_board(board)

        if turn == game.AI and not game_over:

            col, minimax_score = game.AlphaBeta(board, 5, -math.inf, math.inf, True)

            if game.is_valid_location(board, col):
                row = game.get_next_open_row(board, col)
                game.drop_piece(board, row, col, game.AI_PIECE)

                if game.winning_move(board, game.AI_PIECE):
                    label = myfont.render("Player AI wins!", 1, game.COLOR_YELLOW)
                    print(game.WINNING_POSITION)
                    game.screen.blit(label, (40, 10))
                    game_over = True

                # game.print_board(board)
                game.draw_board(board)

                turn += 1
                turn = turn % 2

        if game_over:
            pygame.time.wait(3000)
            return game.WINNING_POSITION
#endregion


# region KMeans_Main_Fn
def Kmeans_Main():
    dataset = DataItem.getDataset()
    # 1 for Euclidean and 0 for Manhattan
    clustering = Clustering_kmeans(dataset, 2, len(dataset), 1)
    clusters = clustering.getClusters()
    for cluster in clusters:
        for i in range(4):
            cluster.centroid[i] = round(cluster.centroid[i], 2)
        print(cluster.centroid[:4])
    return clusters

# endregion


######################## MAIN ###########################33
if __name__ == '__main__':

    SearchAlgorithm_Main()
    Gaming_Main()
    Kmeans_Main()
