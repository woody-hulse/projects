import random

for _ in range(10000):
    sides = [[], [], []]
    for i in range(9):
        added = False
        while not added:
            side = random.randint(0, 2)
            if len(sides[side]) < 3:
                sides[side].append(i + 1)
                added = True

    if sum(sides[0] + [sides[1][0]]) == sum(sides[1] + [sides[2][0]]) == sum(sides[2] + [sides[0][0]]):
        print(sides[0] + [sides[1][0]])
        print(sides[1] + [sides[2][0]])
        print(sides[2] + [sides[0][0]])
        print()