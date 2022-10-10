import random
import os
os.system("clear")

pickValues = {}
with open("pickValues.txt", "r") as fin:
    for line in fin.readlines():
        pick, value, pct = line.split()
        pickValues[int(pick)] = float(value)

def getValue(picks):
    totalValue = 0
    for thisPick in picks:
        try:
            totalValue += pickValues[thisPick]
        except:
            continue
    return totalValue

myPicks = [13, 14, 15, 16, 98, 99, 100, 126, 127, 153, 154, 155, 182, 183, 210, 211]

opponentTeams = {
    0 : "canyon",
    1 : "alex",
    2 : "charlie",
    3 : "leo",
    4 : "lewis",
    5 : "carson",
    6 : "jonji",
    7 : "noah",
    8 : "luke",
    9 : "corey",
    10 : "danny",
    11 : "ryan",
    12 : "jude"
}
opponentTeamPicks = [
    [9, 20, 37, 48, 65, 76, 93, 104, 121, 132, 149, 160, 177, 188, 205, 216],
    [5, 24, 33, 52, 61, 80, 89, 108, 117, 136, 145, 164, 173, 192, 201, 220],
    [8, 21, 53, 60, 75, 81, 94, 103, 109, 122, 131, 159, 178],
    [1, 28, 29, 56, 57, 84, 85, 112, 113, 140, 141, 168, 169, 196, 197, 224],
    [6, 23, 34, 51, 62, 79, 90, 107, 118, 135, 146, 163, 174, 191, 202, 219],
    [3, 26, 31, 54, 55, 82, 87, 110, 115, 138, 143, 166, 171, 194, 199, 222],
    [12, 17, 40, 45, 68, 73, 96, 101, 124, 129, 152, 157, 180, 185, 208, 213],
    [11, 18, 39, 46, 67, 74, 95, 102, 123, 130, 151, 158, 179, 186, 207, 214],
    [4, 7, 10, 66, 92, 116, 133, 137, 144, 150, 165, 172, 200, 206, 215, 217, 221],
    [41, 42, 43, 44, 69, 70, 71, 72, 97, 125, 128, 156, 181, 184, 209, 212],
    [8, 32, 36, 49, 64, 77, 88, 92, 105, 120, 148, 161, 176, 189, 193, 204],
    [2, 27, 30, 55, 58, 83, 86, 111, 114, 139, 142, 167, 170, 195, 198, 223],
    [22, 25, 35, 38, 50, 63, 78, 91, 106, 119, 134, 147, 162, 175, 190, 203, 218]
]

myValue = getValue(myPicks)
opponentTeamPicks[9].append(70)
opponentTeamValues = [getValue(picks) for picks in opponentTeamPicks]

print("\nthis trade :", round(getValue([13, 16, 100, 125]), 3), round(getValue([42, 43, 70, 71]), 3))

print()
for key, name in opponentTeams.items():
    print("{:8} : {:^8} : {:<8}".format(key, name, round(opponentTeamValues[key], 3)))
print()
print("{:8} : {:^8} : {:<8}".format(" ", "me", round(myValue, 3)))
print()

selectedTeam = int(input("team : "))
numPicks = 3
iterationLimit = 10000
requestedTrades = 25
print()
if len(opponentTeamPicks[selectedTeam]) > 0:

    myPickLength = len(myPicks)
    opponentPickLength = len(opponentTeamPicks[selectedTeam])

    iteration = 0
    foundTrades = 0
    while iteration < iterationLimit and foundTrades < requestedTrades:
        iteration += 1
        myTradePicks = []
        opponentTradePicks = []

        for pickExchange in range(numPicks):
            foundPicks = False
            while not foundPicks:
                myPick = myPicks[random.randint(0, myPickLength - 1)]
                opponentPick = opponentTeamPicks[selectedTeam][random.randint(0, opponentPickLength - 1)]
                if myPick not in myTradePicks and opponentPick not in opponentTradePicks:
                    myTradePicks.append(myPick)
                    opponentTradePicks.append(opponentPick)
                    foundPicks = True

        net = getValue(opponentTradePicks) - getValue(myTradePicks)
        if 6 < net < 11:
            foundTrades += 1
            myTradePicks.sort()
            opponentTradePicks.sort()
            print("trade {:3} : {:>26} for {:<20}      team value : {:>5} -> {:<5}".format(
                foundTrades, str(myTradePicks), str(opponentTradePicks), round(myValue, 3), round(myValue + net, 3)))

    print()