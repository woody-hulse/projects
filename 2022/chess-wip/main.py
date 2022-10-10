import os
import sys
import math
import random
from random import randint
from copy import copy, deepcopy


# const data
sides = ['w', 'b']
fullSides = ['white', 'black']
emptySquares = ["W ", "B "]
noCastle = [[False, False], [False, False]]

SQUAREINDEX = {
    "8": 0,
    "7": 1,
    "6": 2,
    "5": 3,
    "4": 4,
    "3": 5,
    "2": 6,
    "1": 7,

    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}

RANKNAME = {
    0: "8",
    1: "7",
    2: "6",
    3: "5",
    4: "4",
    5: "3",
    6: "2",
    7: "1"
}

FILENAME = {
    0: "a",
    1: "b",
    2: "c",
    3: "d",
    4: "e",
    5: "f",
    6: "g",
    7: "h"
}

ELEMENTS = {
    'W ': ' ', #u'\u25FB', # ◼
    'wp': u'\u265F', # ♟
    'wr': u'\u265C', # ♜
    'wk': u'\u265E', # ♞
    'wb': u'\u265D', # ♝
    'wK': u'\u265A', # ♚
    'wq': u'\u265B', # ♛

    'B ': u'\u00B7', # '.',#u'\u25FC', # ◻
    'bp': u'\u2659', # ♙
    'br': u'\u2656', # ♖
    'bk': u'\u2658', # ♘
    'bb': u'\u2657', # ♗
    'bK': u'\u2654', # ♔
    'bq': u'\u2655'  # ♕
}

POINTVALUES = {
    'p': 1,
    'k': 3,
    'b': 3.25,
    'r': 5,
    'q': 9,
    'K': 0
}

sampleBoard = " \
♜ ♞ ♝ ♚ ♛ ♝ ♞ ♜ \n \
♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟ \n \
◼ ◻ ◼ ◻ ◼ ◻ ◼ ◻ \n \
◻ ◼ ◻ ◼ ◻ ◼ ◻ ◼ \n \
◼ ◻ ◼ ◻ ◼ ◻ ◼ ◻ \n \
◻ ◼ ◻ ◼ ◻ ◼ ◻ ◼ \n \
♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙ \n \
♖ ♘ ♗ ♔ ♕ ♗ ♘ ♖ \n"

startingBoard = [
    ["br", "bk", "bb", "bq", "bK", "bb", "bk", "br"],
    ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp"],
    ["W ", "B ", "W ", "B ", "W ", "B ", "W ", "B "],
    ["B ", "W ", "B ", "W ", "B ", "W ", "B ", "W "],
    ["W ", "B ", "W ", "B ", "W ", "B ", "W ", "B "],
    ["B ", "W ", "B ", "W ", "B ", "W ", "B ", "W "],
    ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp"],
    ["wr", "wk", "wb", "wq", "wK", "wb", "wk", "wr"]
]

emptyBoard = [
    ["W ", "B ", "W ", "B ", "W ", "B ", "W ", "B "],
    ["B ", "W ", "B ", "W ", "B ", "W ", "B ", "W "],
    ["W ", "B ", "W ", "B ", "W ", "B ", "W ", "B "],
    ["B ", "W ", "B ", "W ", "B ", "W ", "B ", "W "],
    ["W ", "B ", "W ", "B ", "W ", "B ", "W ", "B "],
    ["B ", "W ", "B ", "W ", "B ", "W ", "B ", "W "],
    ["W ", "B ", "W ", "B ", "W ", "B ", "W ", "B "],
    ["B ", "W ", "B ", "W ", "B ", "W ", "B ", "W "]
]


def nextTurn(turn):
    """
    returns binary inverse
    :param turn:        current turn
    :return:            binary inverse of current turn
    """

    return (turn + 1) % 2


def indicesToMove(indices):
    """
    converts indices to filerank moves
    :param indices:     indices of move
    :return:            filerank name
    """

    return FILENAME[indices[1]] + RANKNAME[indices[0]]

###########


def createVisualBoard(board):
    """
    creates visual version of board from gameBoard
    :param board:       current game board
    :return:            [board] visual representation of board
    """

    # copy board and change elements into utf chars
    visualBoard = [board[i].copy() for i in range(len(board))]
    for i in range(8):
        for j in range(8):
            visualBoard[i][j] = ELEMENTS[visualBoard[i][j]]

    # add ranks and files
    for i in range(8):
        visualBoard[i] = [str(8 - i)] + [' '] + visualBoard[i] + [' '] + [str(8 - i)]
    files = [u'\u25A0', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', ' ', u'\u25A0']
    space = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    visualBoard = [files] + [space] + visualBoard + [space] + [files]

    return visualBoard


def orientVisualBoard(visualBoard, turn):
    """
    orients board based on turn
    :param visualBoard:     current visual board
    :param turn:            current game turn
    :return:                [board] correctly oriented board
    """

    if turn == 1:
        size = len(visualBoard)
        flippedVisualBoard = []
        for i in range(size-1, -1, -1):
            flippedVisualBoard.append([])
            for j in range(size-1, -1, -1):
                flippedVisualBoard[-1].append(visualBoard[i][j])
        visualBoard = flippedVisualBoard

    return visualBoard

def printBoard(board, turn):

    """
    prints a board with createVisualBoard function call
    :param board:       board to print
    :param turn:        game turn or orientation
    :return:            none
    """

    visualBoard = createVisualBoard(board)
    visualBoard = orientVisualBoard(visualBoard, turn)
    r = len(visualBoard)
    c = len(visualBoard[0])

    for i in range(r):
        for j in range(c):
            print(visualBoard[i][j], end=' ')
        print()




#########################


def getKingPosition(board, turn):

    """
    gets the positioning of a side's king
    :param board:       current game board
    :param turn:        side of king
    :return:            i, j of king
    """

    for i in range(8):
        for j in range(8):
            if board[i][j] == sides[turn] + "K":
                return i, j

    return 0, 0


def checkSquare(board, turn, i, j, capture=True):
    """
    returns whether square is available
    :param board:       current board
    :param turn:        game turn
    :param i:           row index
    :param j:           col index
    :param capture      can moving piece capture
    :param onlyCapture  if capturing is only legal move [pawns]
    :return:            [bool] space available
    """

    # pawn diagonal capture
    # if onlyCapture and board[row][col][0] == sides

    # outside of board
    if i < 0 or i > 7 or j < 0 or j > 7:
        return False

    if board[i][j][0] == sides[turn]:
        return False

    # empty square
    if board[i][j][1] == ' ':
        return True

    # capture enemy piece
    if board[i][j][0] == sides[nextTurn(turn)] and capture:
        return True

    return False


def addMove(board, turn, legalMoves, move):
    """
    adds a move if removes check [if applicable]
    :param board:       current board
    :param turn:        player turn
    :param legalMoves:  list of legal moves
    :param move:        suggested move
    :return:            list of legal moves +/- move
    """

    testBoard = makeMove(deepcopy(board), move)
    if len(board) == 9 or not check(testBoard, turn):
        legalMoves.append(move)

    return legalMoves


def getLegalPawnMoves(board, turn, i, j, direction):
    legalMoves = []
    if i == int(-5 * (turn - 0.5) + 3.5):
        if checkSquare(board, turn, i + 2 * direction, j, False) and checkSquare(board, turn, i + direction, j, False):
            move = [[i, j], [i + 2 * direction, j]]
            legalMoves = addMove(board, turn, legalMoves, move)

    if checkSquare(board, turn, i + direction, j, False):
        move = [[i, j], [i + direction, j]]
        legalMoves = addMove(board, turn, legalMoves, move)

    for n in [-1, 1]:
        move = [[i, j], [i + direction, j + n]]
        legalMoves = addMove(board, turn, legalMoves, move)

    # TODO: en passant

    return legalMoves


def getLegalRookMoves(board, turn, i, j):
    legalMoves = []
    for m in [-1, 1]:
        i2 = i
        while 0 <= i2 < 8:
            i2 += m
            if checkSquare(board, turn, i2, j):
                move = [[i, j], [i2, j]]
                legalMoves = addMove(board, turn, legalMoves, move)
                if not board[i2][j][1] == ' ':
                    break
            else:
                break
    for n in [-1, 1]:
        j2 = j
        while 0 <= j2 < 8:
            j2 += n
            if checkSquare(board, turn, i, j2):
                move = [[i, j], [i, j2]]
                legalMoves = addMove(board, turn, legalMoves, move)
                if not board[i][j2][1] == ' ':
                    break
            else:
                break
    return legalMoves


def getLegalBishopMoves(board, turn, i, j):
    legalMoves = []
    for m in [-1, 1]:
        for n in [-1, 1]:
            i2, j2 = i, j
            while 0 <= i2 < 8 and 0 <= j2 < 8:
                i2 += m
                j2 += n
                if checkSquare(board, turn, i2, j2):
                    move = [[i, j], [i2, j2]]
                    legalMoves = addMove(board, turn, legalMoves, move)
                    if not board[i2][j2][1] == ' ':
                        break
                else:
                    break
    return legalMoves


def getLegalKnightMoves(board, turn, i, j):
    legalMoves = []
    for m in [-2, -1, 1, 2]:
        for n in [-2, -1, 1, 2]:
            if abs(m * n) == 2:
                if checkSquare(board, turn, i + m, j + n):
                    move = [[i, j], [i + m, j + n]]
                    legalMoves = addMove(board, turn, legalMoves, move)
    return legalMoves


def getLegalQueenMoves(board, turn, i, j):
    return getLegalBishopMoves(board, turn, i, j) + getLegalRookMoves(board, turn, i, j)


def getLegalKingMoves(board, turn, i, j):
    legalMoves = []
    for m in [-1, 0, 1]:
        for n in [-1, 0, 1]:
            if not (m == 0 and n == 0):
                if checkSquare(board, turn, i + m, j + n):
                    move = [[i, j], [i + m, j + n]]
                    legalMoves = addMove(board, turn, legalMoves, move)
    return legalMoves

def getCastlingMoves(board, turn, i, j, side):

    """
    check viability of castle
    :param board:                   current game board
    :param turn:                    player turn
    :param i:                       rank of king
    :param j:                       file of king
    :param side:                    castling side
    :return:                        legal castle
    """

    attackedSquares = getLegalMoves(board, turn)

    if not (j == 4 and i == 7 - turn * 7):
        return []

    if side == 0:
        for n in [1, 2]:
            if board[i][j + n][1] != ' ' or [i, j + n] in attackedSquares:
                return []
        return [[[i, j], [i, j + 2]]]
    else:
        for n in [-1, -2, -3]:
            if board[i][j + n][1] != ' ' or [i, j + max(n, -2)] in attackedSquares:
                return []
        return [[[i, j], [i, j - 2]]]



def getLegalMoves(board, turn, castle):
    """
    returns list of legal moves for side
    :param board:                   current game board
    :param turn:                    player turn
    :param castle:                  castling viability of both sides
    :return:                        [3DList] of pairs of [[[moveFromRow, moveFromCol], [moveToRow, moveToCol]], ...]
    """

    legalMoves = []

    # direction that pawns move
    direction = 2 * turn - 1

    for i in range(8):
        for j in range(8):
            if board[i][j][0] == sides[turn]:

                # empty square
                if board[i][j][1] == ' ':
                    continue

                # pawn
                if board[i][j][1] == 'p':
                    legalMoves += getLegalPawnMoves(board, turn, i, j, direction)
                # knight
                if board[i][j][1] == 'k':
                    legalMoves += getLegalKnightMoves(board, turn, i, j)
                # bishop
                if board[i][j][1] == 'b':
                    legalMoves += getLegalBishopMoves(board, turn, i, j)
                # rook
                if board[i][j][1] == 'r':
                    legalMoves += getLegalRookMoves(board, turn, i, j)
                # queen
                if board[i][j][1] == 'q':
                    legalMoves += getLegalQueenMoves(board, turn, i, j)
                # king
                if board[i][j][1] == 'K':
                    legalMoves += getLegalKingMoves(board, turn, i, j)

                    for side in range(len(castle[turn])):
                        if castle[turn][side]:
                            legalMoves += getCastlingMoves(board, turn, i, j, side)


    return legalMoves


#############################


def evaluatePosition(board):

    """
    positional evaluation

    :param board:       current game board
    :return:            [-1, 1] positional evaluation
    """

    # white, black
    points = [0, 0]


    """
    bishopCounts = [0, 0]
    developmentScores = [0, 0]
    for i in range(8):
        for j in range(8):
            if board[i][j][0] == 'w':
                points[0] += POINTVALUES[board[i][j][1]]
                if board[i][j][1] == 'b':
                    bishopCounts[0] += 1
            elif board[i][j][0] == 'b':
                points[1] += POINTVALUES[board[i][j][1]]
                if board[i][j][1] == 'b':
                    bishopCounts[1] += 1


            if board[i][j][1] == 'w':
                if board[i][j][0] in ['k', 'b']:
                    developmentScores[0] += 1
            elif board[i][j][1] == 'b':
                if board[i][j][0] in ['k', 'b']:
                    developmentScores[1] += 1


    for turn in [0, 1]:

        points[turn] += developmentScores[turn] / 2

        if bishopCounts[turn] == 2:
            points[turn] += 1

        attackedSquares = getLegalMoves(board, turn)
        points[turn] += len(attackedSquares) / 32
        keySquares = [[3, 3], [3, 4], [4, 3], [4, 4]]

        centerBonus = 0
        for square in attackedSquares:
            if square in keySquares:
                centerBonus += 1
        points[turn] += centerBonus / 4

        kingProtection = 0
        i, j = getKingPosition(board, turn)
        if turn == 0 and i != 7:
            kingProtection -= 4
        elif turn == 1 and i != 0:
            kingProtection -= 4

        for m in [-1, 0, 1]:
            for n in [-1, 0, 1]:
                if i + m < 0 or i + m >= 8 or j + n < 0 or j + n >= 8:
                    kingProtection += 2
                elif board[i + m][j + n][0] == sides[turn]:
                    kingProtection += 1
                elif board[i + m][j + n][0] == sides[nextTurn(turn)]:
                    kingProtection -= 5
        kingProtection /= 8
        points[turn] += kingProtection
    """

    difference = points[0] - points[1]

    evaluation = 1 / (1 + pow(math.e, difference * -0.2))

    return random.random()

    # return evaluation


def printEvaluationBar(board, evaluation=-1):
    """
    print a visual evaluation meter
    :param board:       current game board
    :param evaluation   given evaluation
    :return:            none
    """

    if evaluation == -1:
        evaluation = evaluatePosition(board)
    barlen = len(board) * 3 - 3
    fill = barlen * evaluation

    print(' ', end='')
    for barIndex in range(barlen):
        if barIndex <= fill:
            print('■', end='')
        else:
            print('□', end='')
    print()


def check(board, turn):

    """
    returns if player is in check
    :param board:           current game board
    :param turn:            player turn
    :return:                [bool] checkmate
    """
    moves = getLegalMoves(board, turn, noCastle)
    ki, kj = getKingPosition(board, turn)
    if [ki, kj] in moves:
        return True

    return False


def avoidsCheck(board, turn, move):
    """
    returns if player is still in check
    :param board:       current game board
    :param turn:        player turn
    :param move         tested move
    :return:            [bool] checkmate
    """

    newBoard = makeMove(deepcopy(board), move)
    return not check(newBoard, turn)


def minimax(board, turn, castle, abp, depth=0, maxDepth=1):

    """
    recursive algorithm to determine optimal move
    :param board:       game board
    :param turn:        player turn
    :param castle:      castling viability of each side
    :param abp:         alpha-beta pruning value [val, optimal]
    :param depth:       depth of search
    :param maxDepth:    maximum allowable depth of search
    :return:            move score or best move and evaluation
    """

    if depth == maxDepth:
        return evaluatePosition(board)

    legalMoves = getLegalMoves(board, turn, castle)

    if len(legalMoves) == 0:
        if check(board, turn):
            return nextTurn(turn)
        else:
            return 0.5

    moveScores = []
    bestScore = turn # alpha-beta pruning
    bestMove = [[], []]

    for move in legalMoves:

        score = minimax(makeMove(deepcopy(board), move), nextTurn(turn),
                        findCastlingViability(board, turn, move, deepcopy(castle)),
                        [bestScore, nextTurn(turn)],
                        depth + 1,
                        maxDepth)
        if abs(score - nextTurn(turn)) < abs(bestScore - nextTurn(turn)):
            bestScore = score
            bestMove = move
        moveScores.append(score)

    if depth == 0:
        # TODO: if len(bestIndices) == 0, stalemate
        return bestMove, bestScore
    else:
        return bestScore


#############################

def getMove(legalMoves):
    """
    collects a player's move
    :param legalMoves   list of legal moves for player
    :return:            [2DList] of valid [[moveFromRow, moveFromCol], [moveToRow, moveToCol]]
    """

    chosen = False

    while not chosen:
        print()
        moveFrom, moveTo = input(": ").split()

        try:
            i1, j1 = SQUAREINDEX[moveFrom[1]], SQUAREINDEX[moveFrom[0]]
            i2, j2 = SQUAREINDEX[moveTo[1]], SQUAREINDEX[moveTo[0]]
            move = [[i1, j1], [i2, j2]]
        except:
            print("invalid square selection")
            continue

        if move in legalMoves:
            return move
        else:
            print("invalid move")


def findCastlingViability(board, turn, move, castle):

    """
    finds which castles are ineligible after move
    :param board:       current game board
    :param turn:        player turn
    :param move:        2DList of pairs
    :param castle:      viability of castling
    :return:            castling viability
    """

    if castle[turn] == [False, False]:
        return castle

    piece = board[move[0][0]][move[0][1]]
    if piece[1] == 'K':
        castle[turn] = [False, False]
    else:
        if [7, 7] in move:
            castle[0][0] = False
        elif [7, 0] in move:
            castle[0][1] = False
        if [0, 7] in move:
            castle[1][0] = False
        elif [0, 0] in move:
            castle[1][1] = False

    return castle

def makeMove(board, move, autoQueen=True):
    """
    returns board with moved piece
    :param board:       current game board
    :param move:        2DList of pairs
    :param autoQueen    automatically queen promotions
    :return:            new board
    """

    piece = board[move[0][0]][move[0][1]]

    # castle
    if piece[1] == 'K' and abs(move[1][1] - move[0][1]) == 2 and move[0][1] == 4:
        side = move[1][0] - move[0][0]
        if side == -2:
            rook = board[move[0][0]][move[0][1] - 4]
            board[move[0][0]][move[0][1] - 4] = emptySquares[(move[0][0] + move[0][1] - 4) % 2]
            board[move[0][0]][move[0][1] - 1] = rook
        else:
            rook = board[move[0][0]][move[0][1] + 3]
            board[move[0][0]][move[0][1] + 3] = emptySquares[(move[0][0] + move[0][1] + 3) % 2]
            board[move[0][0]][move[0][1] + 1] = rook
        board[move[1][0]][move[1][1]] = piece

        board[move[0][0]][move[0][1]] = emptySquares[(move[0][0] + move[0][1]) % 2]

    else:
        board[move[0][0]][move[0][1]] = emptySquares[(move[0][0] + move[0][1]) % 2]
        board[move[1][0]][move[1][1]] = piece

        # pawn promotion
        if piece[1] == 'p' and (move[1][0] == 0 or move[1][0] == 7):
            if autoQueen:
                pieceType = 'q'
            else:
                pieceType = ''
                while pieceType not in ['k','b','r','q']:
                    pieceType = input("select promotion piece: ")
            board[move[1][0]][move[1][1]] = piece[0] + pieceType

    return board



##############################


def playGame():
    # defaults
    gameEnd = False
    board = startingBoard

    turn = 0
    ai = [False, True]
    aiDepth = 3

    castle = [[True, True], [True, True]]

    while not gameEnd:

        if not ai[turn]:

            os.system("clear")

            printBoard(board, turn)
            print()
            printEvaluationBar(board)

            legalMoves = getLegalMoves(board, turn, castle)

            if check(board, turn):
                print()
                if len(legalMoves) == 0:
                    print(fullSides[turn], "has been checkmated")
                    break
                else:
                    print(fullSides[turn], "is in check")
            elif len(legalMoves) == 0:
                print()
                print("stalemate")
                break

            # print("best:", indicesToMove(bestMove[0]), indicesToMove(bestMove[1]))

            move = getMove(legalMoves)
            board = makeMove(board, move, False)

        else:
            abp = [nextTurn(turn), turn]
            if ai == [True, True]:
                os.system("clear")
                printBoard(board, turn)
                bestMove, evaluation = minimax(board, turn, castle, abp, 0, aiDepth)
                printEvaluationBar(board)
            else:
                bestMove, evaluation = minimax(board, turn, castle, abp, 0, aiDepth)

            move = bestMove
            board = makeMove(board, move)


        castle = findCastlingViability(board, turn, move, castle)

        turn = nextTurn(turn)

def main():

    playGame()


if __name__ == "__main__":
    main()