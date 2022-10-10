startFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

class Piece:
    none = 0
    pawn = 1
    knight = 2
    bishop = 3
    rook = 4
    queen = 5
    king = 6

    white = 8
    black = 16

class Board:

    squares = [0 for _ in range(64)]

    def loadBoardFromFEN(self, fen):
        pieceDefinition = {
            'p': Piece.pawn,
            'n': Piece.knight,
            'b': Piece.bishop,
            'r': Piece.rook,
            'q': Piece.queen,
            'k': Piece.king
        }

        fenBoard = fen.split(' ')[0]

        file = 0
        rank = 7

        for char in fenBoard:

            if char == '/':
                file = 0
                rank -= 1
            else:
                if char.isdigit():
                    file += int(char)
                else:
                    pieceType = pieceDefinition[char.lower()]
                    if char.isupper():
                        pieceColor = Piece.white
                    else:
                        pieceColor = Piece.black
                    self.squares[rank * 8 + file] = bin(pieceType | pieceColor)
                    file += 1

board = Board()
board.loadBoardFromFEN(startFEN)
print(board.squares)