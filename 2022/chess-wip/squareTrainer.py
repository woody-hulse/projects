import math
import random
import os
import time

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

def showSquare(label):
    board = [[' ' if (i + j) % 2 == 0 else u'\u00B7' for j in range(8)] for i in range(8)]

    m, n = random.randint(0, 7), random.randint(0, 7)
    board[m][n] = '0'

    for i in range(8):
        if label:
            board[i] = [str(8 - i)] + [' '] + board[i] + [' '] + [str(8 - i)]
        else:
            board[i] = [' ', ' '] + board[i] + [' ', ' ']

    if label:
        files = [u'\u25A0', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', ' ', u'\u25A0']
    else:
        files = [u'\u25A0', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', u'\u25A0']
    space = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    board = [files] + [space] + board + [space] + [files]

    for i in range(len(board)):
        for j in range(len(board[i])):
            print(board[i][j], end=' ')
        print()
    print()

    return FILENAME[n] + RANKNAME[m]

label = True
start = time.time()
correct, attempted = 0, 0
while True:

    os.system("clear")
    try:
        print("\n", correct, "/", attempted, " ", int((time.time() - start) / correct), "sec / correct \n")
    except ZeroDivisionError:
        print("\n")

    square = showSquare(label)
    answer = input("\n : ")

    if answer.lower() == "label":
        label = True
        continue
    elif answer.lower() == "nolabel":
        label = False
        continue

    correct += 1 if answer == square else 0
    attempted += 1
