#include <iostream>
#include <algorithm>
#include <iterator>
#include <string>
#include <list>
#include <random>
using namespace std;

int win = -1;

int wins, losses, draws = 0;

string clrscr = "\n\n\n\n\n";

// whose turn is it
string players[] = {"x", "o", "-"};
int turn = 0;

int rows = 6;
int cols = 7;
string board[6][7];

string createBoard(string thisBoard[6][7]) {
	string display = "";

	// insert column labels
	for (int i = 0; i < cols; i++) {
		display += "  " + to_string(i + 1) + " ";
	}

	display += "\n";

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j <= cols * 2; ++j) {

			int k = (j - 1) / 2;

			if (j % 2 == 0) {
				display += "|";
			} else {
				display += " " + thisBoard[i][k] + " ";
			}
		}
		display += "\n";
	}

	return display;
}

bool checkFourInARow(string arr[], string player) {

	string displayed = "";

	for (int i = 0; i < cols*2; i++) {
		displayed += arr[i] + " ";
	}

	int numConsecutive = 0;

	for (int i = 0; i < cols * 2; i++) {
		if (arr[i] == player) numConsecutive += 1;
		else numConsecutive = 0;

		if (numConsecutive == 4) return true;
	}

	return false;
}


bool checkWin(string thisBoard[6][7], int row, int col, string player) {

	string arr1[cols * 2], arr2[cols * 2], arr3[cols * 2], arr0[cols * 2];
	for (int i = -1 * cols + 1; i < cols; i++) {
		if (col + i >= 0 and col + i < cols) {
			if (row + i >= 0 and row + i < rows)
				arr1[i + cols] = thisBoard[row + i][col + i];
			if (row - i >= 0 and row - i < rows)
				arr2[i + cols] = thisBoard[row - i][col + i];

			arr3[i + cols] = thisBoard[row][col + i];
		}
	}

	for (int i = 0; i < rows; i++) {
		arr0[i] = thisBoard[i][col];
	}

	if (checkFourInARow(arr0, player) or
		checkFourInARow(arr1, player) or
		checkFourInARow(arr2, player) or
		checkFourInARow(arr3, player)) return true;

	return false;
}

bool checkFull(string thisBoard[6][7]) {

	for (int i = 0; i < cols; i++)
		if (thisBoard[0][i] == " ")
			return false;

	return true;
}

// return -2 means failed move, -1 means moved, 0 is win for x, 1 is win for y, 2 is draw
int move(string thisBoard[6][7], string player, int col) {

	for (int i = 0; i < rows; i++) {
		if (thisBoard[i][col] == " " and (i == rows - 1 or thisBoard[i+1][col] != " ")) {
			thisBoard[i][col] = player;

			if (i == 0)
				if (checkFull(thisBoard))
					return 2;

			if (checkWin(thisBoard, i, col, player)) {
				if (player == "x") return 0;
				else return 1;
			}
			return -1;
		}
	}
	return -2;
}

void printArray(float arr[7]) {
	string display = "";

	for (int i = 0; i < 7; i++) {
		display += " " + to_string(arr[i]);
	}

	cout << display << endl;
}

mt19937 gen( chrono::system_clock::now().time_since_epoch().count() );
template <class T > void shuffleList( list<T> &L )
{
   vector<T> V( L.begin(), L.end() );
   shuffle( V.begin(), V.end(), gen );
   L.assign( V.begin(), V.end() );
}


float minimax(string thisBoard[6][7], int playerIndex, int depth, int maxDepth) {

	if (depth > maxDepth)
		return 0;

	float possibleMoves[7];
	string newBoard[7][6][7];

	for (int i = 0; i < cols; i++) {

		memcpy(newBoard[i], thisBoard, 7*6*sizeof(string));

		int moveVal = move(newBoard[i], players[playerIndex], i);

		if (moveVal == turn and depth == 0) win = 1;

		if (moveVal == turn) possibleMoves[i] = 1 - (depth / 100.0);
		if (moveVal == (turn + 1) % 2) possibleMoves[i] = -1 + (depth / 100.0);
		if (moveVal == -2) possibleMoves[i] = -4;
		if (moveVal == -1) {
			possibleMoves[i] = minimax(newBoard[i], (playerIndex + 1) % 2, depth + 1, maxDepth);
		}
	}

	if (depth == 0) printArray(possibleMoves);

	float extremeVal = 0;
	list<float> bestMoves;
	if (playerIndex == turn) {
		extremeVal = -3;
		for (int i = 0; i < cols; i++) {
			if (possibleMoves[i] > extremeVal and possibleMoves[i] != -4) {
				extremeVal = possibleMoves[i];
				bestMoves.clear();
				bestMoves.push_front(i);
			} else if (possibleMoves[i] == extremeVal) bestMoves.push_back(i);
		}
	} else {
		extremeVal = 3;
		for (int i = 0; i < cols; i++) {
			if (possibleMoves[i] < extremeVal and possibleMoves[i] != -4) {
				extremeVal = possibleMoves[i];
				bestMoves.clear();
								bestMoves.push_front(i);
			} else if (possibleMoves[i] == extremeVal) bestMoves.push_back(i);
		}
	}

	if (depth == 0) {
		shuffleList(bestMoves);
		int choice = bestMoves.front();
		while (thisBoard[0][choice] != " ") {
			shuffleList(bestMoves);
			choice = bestMoves.front();
		}
		return choice;
	} else return extremeVal;
}

void reset() {
	win = -1;
	turn = 0;
}

void playGame(int gameMode, int difficulty) {

	// create initial board state
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			board[i][j] = " ";
		}
	}

	while (win == -1) {

		cout << createBoard(board) << endl;

		// check if choice is a real column number
		bool placedMarker = false;

		while ((turn == 0 or gameMode == 2) and !placedMarker and gameMode != 0) {

			cout << "Where would you like to move, player " << players[turn] << "? ";

			int choice = 0;
			cin >> choice;


			string failMessage = "Spot not available, choose again\n\n";

			if ((unsigned)(choice - 1) < cols) {
				if (board[0][choice - 1] == " ") {
					int val = move(board, players[turn], choice - 1);
					if (val != -2) {
						placedMarker = true;

						if (val != -1) {
							win = val;
						}
					}
				}
			}

			if (!placedMarker) cout << failMessage;
		}

		if ((turn == 1 and gameMode == 1) or gameMode == 0) {
			move(board, players[turn], floor(minimax(board, turn, 0, difficulty)));
		}

		// switch turn
		turn = (turn + 1) % 2;

		// clear screen
		cout << clrscr;
	}

	cout << createBoard(board) << endl;

	if (gameMode == 1) {
		if (win == 0) wins ++;
		else if (win == 1) losses ++;
		else if (win == 2) draws ++;
	}

	if (win == 2) cout << "It was a draw!" << endl << endl;
	else cout << players[win] << " wins!!!!!!" << endl << endl;
}

int main() {

	while (true) {

		int gameMode = 0;

		cout << "Welcome!" << endl;
		cout << "You are " << wins << "-" << losses << "-" << draws << " against ai opponents" << endl;
		cout << "Would you like to exit (-1) or play (0), (1), or (2) player mode? ";
		cin >> gameMode;

		reset();

		if (gameMode == -1) break;

		int difficulty = 0;
		if (gameMode == 0 or gameMode == 1) {
			cout << "Enter difficulty (1-10): ";
			cin >> difficulty;

			if (gameMode == 1) {
				string firstTurn = "";
				cout << "Who goes first, (player) or (ai)? ";
				cin >> firstTurn;

				if (firstTurn == "ai")
					turn = 1;
			}
		}

		cout << clrscr;
		playGame(gameMode, difficulty);
	}

	return 0;
}
