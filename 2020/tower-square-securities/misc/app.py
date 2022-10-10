# no semicolons lol
# listA, listB are lists (defined by 'listA = [item1, item2]' )
# a is variable
#
# listA.extend(listB)
# listA.append(a)
# listA.insert(1, a)
# listA.remove(a)
# listA.clear()
# listA.pop()       <- removes last element from listA
# listA.index(a)        <- determines index of item a
# listA.count(a)        <- determines how many times a occurs in listA
# listA.sort()      <- for text puts in alphabetical order, numbers in ascending order
# listA.reverse()
# listB = listA.copy()
# len(listA)        <- returns number of items in listA


# tuples are defined by '()' and cannot be changed


# functions defined by 'def name( parameters ):
#   'name' is name of function, 'parameters' are parameters of function
# indent items in functions
# called by name( parameters )
#   'return' can be used in any function, doesn't need 'def' to be changed


# functions defined by 'if condition:' where 'condition' is something like a > 2 (no parentheses)
# 'else:'
# 'elif condition2 and not(condition1):
# 'if condition1 or condition2:' ('or' can be changed to 'and')
# same condition statements as java ('==', '!=', '<', etc)


# dictionary defined by:
#   name = {
#       item1 : definition1
#       item2 : definition2
#   }
# items must be unique
# accessed by 'name[item]' or 'name.get(item)'
# if item is not in dictionary will return "None"
# 'name.get(item, defaultReturn)' will return defaultReturn if item not in dictionary


# 'while condition:'
# same format as 'if' statements


# for letter in String:         <- where 'String' is a string and 'letter' is item in string
# for a in listA:       <- where 'listA' is an array and 'a' is element of array
# for index in range(start, finish)     <- where index is a number from [start, finish) exclusive of finish
#
# def raiseToPower(baseNum, powNum):
#   result = 1
#   for index in range(powNm):
#       result = result * baseNum
#   return result


# 2d array defined by 'name = [ [items1], [items2] ]
#   referenced by name[i][j]
#
# for row in nums:       <- 'nums' is 2d array
#   for col in row:
#       print(col)


# try:
#   *do something*
# except:
#   print("invalid")
#
# 'except' can be followed by error type (ZeroDivisionError, ValueError, etc)
# can have multiple 'except' s


# opening/reading files      <- 'file' is whatever name for the file you want
# file = open("fileName", "function")        <- where 'function' is one of the following terms:
#   "r" - read
#   "w" - write
#   "a" - append
#   "r+" - read and write
# file.readable()       <- returns boolean value for if file is readable, will be false if function is not "r"
# file.read()       <- will return all information inside file
# file.readline()       <- will read next line (could be repeated and read down the whole file)
# file.readlines()      <- will create an array of all lines in file (better than 'readline()' )
# file.close()       <- necessary to close file after opening
#
# for item in file.readlines():
#   print(item)
#
# if function "a" is chosen:
# file.write(item)      <- adds this item to end of file (add \n before item to write entry to new line)
#
# if function "w" is chosen:
# file.write("item")      <- will overwrite all preexisting items in 'file'
# newFile = open("fileName", "function")        <- 'fileName' has .html to define it as HTML file
# newFile.write("< >")       <- '< >' can be any HTML script
#
# import fileName       <- will import all functions from 'fileName'
#
# fileName.function     <- wll run 'function' in 'fileName'
# fileName.variable     <- will return 'variable' in 'fileName'

# https://docs.python.org/3/py-modindex.html  <- for Python modules which can be accessed (stored in External Libraries)
#
# 'pip' is a package manager which allows you to download third-party Python modules

# from math import *

import pygame
import random

pygame.font.init()

width = 800
height = 700
game_width = 300
game_height = 600
block_size = game_width / 10

top_left_x = (width - game_width) // 2
top_left_y = height - game_height


# shapes

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]


class Shape(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0


def create_grid(locked_pos={}):
    grid = [[(0,0,0) for _ in range(10)] for _ in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_pos:
                c = locked_pos[(j,i)]
                grid[i][j] = c
    return grid


def convert_shape_format(shape):
    positions = []
    form = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(form):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == (0, 0, 0)] for i in range(20)]
    accepted_pos = [j for sub in accepted_pos for j in sub]

    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_pos:
            if pos[1] > -1:
                return False
    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True

    return False


def get_shape():
    return Shape(5, 0, random.choice(shapes))


def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont("menlo", size)
    label = font.render(text, 1, color)

    surface.blit(label, (int(top_left_x + game_width / 2 - (label.get_width() / 2)), int(top_left_y + game_height / 2 - label.get_height() / 2)))


def draw_grid(surface, grid):
    sx = top_left_x
    sy = top_left_y

    for i in range(len(grid)):
        pygame.draw.line(surface, (128,128,128), (sx, sy + i*block_size), (sx + game_width, sy+ i*block_size))
        for j in range(len(grid[i])):
            pygame.draw.line(surface, (128, 128, 128), (sx + j*block_size, sy), (sx + j*block_size, sy + game_height))


def clear_rows(grid, locked):

    inc = 0
    for i in range(len(grid)-1, -1, -1):
        row = grid[i]
        if (0,0,0) not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j,i)]
                except:
                    continue

    if inc > 0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                new_key = (x, y + inc)
                locked[new_key] = locked.pop(key)

    return inc


def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('menlo', 30)
    label = font.render('next shape', 1, (255,255,255))

    sx = top_left_x + game_width + 50
    sy = top_left_y + game_height/2 - 100
    form = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(form):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, (sx + j * block_size, sy + i * block_size, block_size, block_size), 0)

    surface.blit(label, (sx + 10, sy - 30))


def update_score(nscore):
    score = max_score()

    with open('scores.txt', 'w') as f:
        if int(score) > nscore:
            f.write(str(score))
        else:
            f.write(str(nscore))


def max_score():
    with open('scores.txt', 'r') as f:
        lines = f.readlines()
        score = lines[0].strip()

    return score


def draw_window(surface, grid, score=0, last_score = 0):
    surface.fill((0, 0, 0))

    pygame.font.init()
    font = pygame.font.SysFont('menlo', 60)
    label = font.render('tetris', 1, (255, 255, 255))

    surface.blit(label, (top_left_x + game_width / 2 - (label.get_width() / 2), 30))

    # current score
    font = pygame.font.SysFont('menlo', 30)
    label = font.render('score: ' + str(score), 1, (255,255,255))

    sx = top_left_x + game_width + 50
    sy = top_left_y + game_height/2 - 100

    surface.blit(label, (sx + 20, sy + 160))
    # last score
    label = font.render('high score: ' + last_score, 1, (255,255,255))

    sx = top_left_x - 200
    sy = top_left_y + 200

    surface.blit(label, (sx + 20, sy + 160))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], (top_left_x + j*block_size, top_left_y + i*block_size, block_size, block_size), 0)

    pygame.draw.rect(surface, (255, 0, 0), (top_left_x, top_left_y, game_width, game_height), 5)

    draw_grid(surface, grid)

    pygame.display.update()


def main(win):
    last_score = max_score()
    locked_positions = {}
    grid = create_grid(locked_positions)

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.27
    level_time = 0
    score = 0

    while run:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        level_time += clock.get_rawtime()
        clock.tick()

        if level_time/1000 > 5:
            level_time = 0
            if level_time > 0.12:
                level_time -= 0.005

        if fall_time/1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not(valid_space(current_piece, grid)) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1
                    if not(valid_space(current_piece, grid)):
                        current_piece.x += 1
                if event.key == pygame.K_RIGHT:
                    current_piece.x += 1
                    if not(valid_space(current_piece, grid)):
                        current_piece.x -= 1
                if event.key == pygame.K_DOWN:
                    current_piece.y += 1
                    if not(valid_space(current_piece, grid)):
                        current_piece.y -= 1
                if event.key == pygame.K_UP:
                    current_piece.rotation += 1
                    if not(valid_space(current_piece, grid)):
                        current_piece.rotation -= 1

        shape_pos = convert_shape_format(current_piece)

        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color

        if change_piece:
            for pos in shape_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False
            score += clear_rows(grid, locked_positions) * 10

        draw_window(win, grid, score, last_score)
        draw_next_shape(next_piece, win)
        pygame.display.update()

        if check_lost(locked_positions):
            draw_text_middle(win, "you lost :(", 80, (255, 255, 255))
            pygame.display.update()
            pygame.time.delay(1500)
            run = False
            update_score(score)


def main_menu(win):
    run = True
    while run:
        win.fill((0, 0, 0))
        draw_text_middle(win, 'press any key to play', 60, (255, 255, 255))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                main(win)

    pygame.display.quit()


win = pygame.display.set_mode((width, height))
pygame.display.set_caption('tetris')
main_menu(win)