

total = -60000
income = 250000
for i in range(50):
    total *= 1.04
    income *= 1.04
    total += income / 2

    print(i + 1, ":", round(total, 2))

print()
print(income)