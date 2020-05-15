# import math
# def minimax(board,nodeindex,depth,isAIPlayer):
#
#       if depth == 0:
#              return board[nodeindex]
#
#       if isAIPlayer:
#          # bestValue = -1000
#          #  value = minimax(board, nodeindex * 2, depth-1, False)
#          #  bestValue = max(value, bestValue)
#          return max(minimax(board, nodeindex * 2, depth - 1, False),minimax(board, nodeindex * 2 + 1, depth - 1,False))
#       else:
#           # bestValue = 1000
#           # value = minimax(board, nodeindex * 2 + 1, depth-1, True)
#           # bestValue = min(bestValue, value)
#           return min(minimax(board,nodeindex * 2, depth - 1, True), minimax(board, nodeindex * 2 + 1, depth - 1,True))
# scores = [3, 5, 2, 9, 12, 5, 23, 23]
# treeDepth = math.log(len(scores), 2)
# print(str(minimax(scores, 0, treeDepth, True)))
MAX , MIN = 1000, -1000

def minimax(depth , nodeIndex,maximizerPlayer,values,alpha,beta):
    if depth == 0:
        return values[nodeIndex]

    if maximizerPlayer:
        bestValue = MIN
        for i in range(0, 2):
             value = minimax(depth - 1, nodeIndex * 2 + i, False, values, alpha, beta)
             bestValue = max(bestValue, value)
             alpha = max(alpha, bestValue)
             if beta <= alpha:
                 break
        return bestValue

    else:
        bestValue = MAX
        for i in range(0, 2):
            value = minimax(depth - 1, nodeIndex * 2 + i, True, values, alpha, beta)
            bestValue = min(bestValue, value)
            beta = min(beta, bestValue)
            if beta <= alpha:
                break
        return bestValue

values = [3, 5, 6, 9, 1, 2, 0, -1]
print("The optimal value is :", minimax(3, 0, True, values, MIN, MAX))
