type = {"♖": "black rook",
        "♘": "black knight",
        "♗": "black bishop",
        "♕": "black queen",
        "♔": "black king",
        "♙": "black pawn",

        "♜": "white rook",
        "♞": "white knight",
        "♝": "white bishop",
        "♛": "white queen",
        "♚": "white king",
        "♟": "white pawn",

        " ": "empty empty"}

col_to_num = {"a": 0,
              "b": 1,
              "c": 2,
              "d": 3,
              "e": 4,
              "f": 5,
              "g": 6,
              "h": 7}

players = ["white", "black"]


def reset_board():
    this_board = []

    black_end_row = ["♖", "♘", "♗", "♕", "♔", "♗", "♘", "♖"]
    black_pawn_row = []

    white_end_row = ["♜", "♞", "♝", "♛", "♚", "♝", "♞", "♜"]
    white_pawn_row = []

    empty_row = []
    for i in range(8):
        black_pawn_row.append("♙")
        white_pawn_row.append("♟")
        empty_row.append(" ")

    this_board.append(black_end_row.copy())
    this_board.append(black_pawn_row.copy())
    for i in range(4):
        this_board.append(empty_row.copy())

    this_board.append(white_pawn_row.copy())
    this_board.append(white_end_row.copy())

    return this_board


def display_board(this_board):

    display = ""

    # column labels
    display += "     a   b   c   d   e   f   g   h   \n"

    for i in range(8):
        for j in range(8):
            # row labels
            if j == 0:
                display += str(8 - i) + " "

            display += " | " + this_board[i][j]

            if j == 7:
                display += " |\n"

    print(display)


def cond(r1, c1, r2, c2):
    return str(r1) + str(c1) + str(r2) + str(c2)


def possible_moves(this_board, color):

    moves = []

    for i in range(8):
        for j in range(8):
            direction = 1
            this_type = type[this_board[i][j]].split()
            if this_type[0] == "white":
                direction = -1

            if this_type[0] == color:
                # pawn
                if this_type[1] == "pawn":
                    for k in range(1, 3):
                        if 0 <= i + k*direction < 8 and \
                                (k == 1 or ((color == "white" and i == 6) or (color == "black" and i == 1))):
                            if type[this_board[i + k*direction][j]].split()[0] != color:
                                moves.append(cond(i, j, i + k*direction, j))
                            else:
                                break
                        lat = int((k - 1.5) * 2)
                        if 0 <= i + k < 8 and 0 <= j + direction < 8:
                            if type[this_board[i+direction][j+lat]].split()[0] != color and \
                                    this_board[i+direction][j+lat] != " ":
                                moves.append(cond(i, j, i + lat, j + direction))

                # rook
                elif this_type[1] == "rook":
                    valid = [True, True, True, True]
                    for k in range(8):
                        if valid[0]:
                            if i + k < 8:
                                if type[this_board[i + k][j]].split()[0] != color:
                                    moves.append(cond(i, j, i + k, j))
                                else:
                                    valid[0] = False
                        if valid[1]:
                            if i - k >= 0:
                                if type[this_board[i - k][j]].split()[0] != color:
                                    moves.append(cond(i, j, i - k, j))
                                else:
                                    valid[1] = False
                        if valid[2]:
                            if j + k < 8:
                                if type[this_board[i][j + k]].split()[0] != color:
                                    moves.append(cond(i, j, i, j + k))
                                else:
                                    valid[2] = False
                        if valid[3]:
                            if j - k >= 0:
                                if type[this_board[i][j - k]].split()[0] != color:
                                    moves.append(cond(i, j, i, j - k))
                                else:
                                    valid[3] = False

                # knight
                elif this_type[1] == "knight":
                    for k in range(-2, 3):
                        for m in range(-2, 3):
                            if abs(k) != abs(m) and k != 0 and m != 0:
                                if 0 <= i + k < 8 and 0 <= j + m < 8:
                                    if type[this_board[i + k][j + m]].split()[0] != color:
                                        moves.append(cond(i, j, i + k, j + m))

                # bishop
                elif this_type[1] == "bishop":
                    valid = [True, True, True, True]
                    for k in range(8):
                        if valid[0]:
                            if i + k < 8 and j + k < 8:
                                if type[this_board[i + k][j + k]].split()[0] != color:
                                    moves.append(cond(i, j, i + k, j + k))
                                else:
                                    valid[0] = False
                        if valid[1]:
                            if i - k >= 0 and j + k < 8:
                                if type[this_board[i - k][j + k]].split()[0] != color:
                                    moves.append(cond(i, j, i - k, j + k))
                                else:
                                    valid[1] = False
                        if valid[2]:
                            if i + k < 8 and j - k >= 0:
                                if type[this_board[i + k][j - k]].split()[0] != color:
                                    moves.append(cond(i, j, i + k, j - k))
                                else:
                                    valid[2] = False
                        if valid[3]:
                            if i - k >= 0 and j - k >= 0:
                                if type[this_board[i - k][j - k]].split()[0] != color:
                                    moves.append(cond(i, j, i - k, j - k))
                                else:
                                    valid[3] = False

                # queen
                elif this_type[1] == "queen":
                    valid = [True, True, True, True, True, True, True, True]
                    for k in range(8):
                        if valid[0]:
                            if i + k < 8:
                                if type[this_board[i + k][j]].split()[0] != color:
                                    moves.append(cond(i, j, i + k, j))
                                else:
                                    valid[0] = False
                        if valid[1]:
                            if i - k >= 0:
                                if type[this_board[i - k][j]].split()[0] != color:
                                    moves.append(cond(i, j, i - k, j))
                                else:
                                    valid[1] = False
                        if valid[2]:
                            if j + k < 8:
                                if type[this_board[i][j + k]].split()[0] != color:
                                    moves.append(cond(i, j, i, j + k))
                                else:
                                    valid[2] = False
                        if valid[3]:
                            if j - k >= 0:
                                if type[this_board[i][j - k]].split()[0] != color:
                                    moves.append(cond(i, j, i, j - k))
                                else:
                                    valid[3] = False
                        if valid[4]:
                            if i + k < 8 and j + k < 8:
                                if type[this_board[i + k][j + k]].split()[0] != color:
                                    moves.append(cond(i, j, i + k, j + k))
                                else:
                                    valid[4] = False
                        if valid[5]:
                            if i - k >= 0 and j + k < 8:
                                if type[this_board[i - k][j + k]].split()[0] != color:
                                    moves.append(cond(i, j, i - k, j + k))
                                else:
                                    valid[5] = False
                        if valid[6]:
                            if i + k < 8 and j - k >= 0:
                                if type[this_board[i + k][j - k]].split()[0] != color:
                                    moves.append(cond(i, j, i + k, j - k))
                                else:
                                    valid[6] = False
                        if valid[7]:
                            if i - k >= 0 and j - k >= 0:
                                if type[this_board[i - k][j - k]].split()[0] != color:
                                    moves.append(cond(i, j, i - k, j - k))
                                else:
                                    valid[7] = False

                # king
                elif this_type[1] == "king":
                    for k in range(-1, 2):
                        for m in range(-1, 2):
                            if 0 <= i + k < 8 and 0 <= j + m < 8:
                                if type[this_board[i + k][j + m]].split()[0] != color:
                                    moves.append(cond(i, j, i + k, j + m))

    return moves


def translate(entry):

    try:
        final = ""
        final += str(8 - int(entry[1]))
        final += str(col_to_num[entry[0]])
        final += str(8 - int(entry[4]))
        final += str(col_to_num[entry[3]])
        return final
    except:
        return ""


def move(entry, this_board, color):

    print(possible_moves(this_board, color))

    if entry in possible_moves(this_board, color):
        r1, c1 = int(entry[0]), int(entry[1])
        r2, c2 = int(entry[2]), int(entry[3])

        this_board[r2][c2], this_board[r1][c1] = this_board[r1][c1], " "

        return this_board

    raise Exception()


def main():

    turn = 0

    win = -1

    board = reset_board()

    while win == -1:

        moved = False

        display_board(board)

        while not moved:

            print(players[turn], "to play")
            print("Where would", players[turn], "like to go? ")

            try:
                board = move(translate(input()), board, players[turn])
                moved = True
            except:
                display_board(board)
                print("Invalid move, try again")
                continue

            turn = (turn + 1) % 2


main()
