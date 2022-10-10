import xlrd

hitters_loc = ("/Users/woodyhulse/Documents/personal/Hitters.xls")
pitchers_loc = ("/Users/woodyhulse/Documents/personal/Pitchers.xls")

hitters_wb = xlrd.open_workbook(hitters_loc)
hitters_sheet = hitters_wb.sheet_by_index(0)
hitters_rows = hitters_sheet.nrows

pitchers_wb = xlrd.open_workbook(pitchers_loc)
pitchers_sheet = pitchers_wb.sheet_by_index(0)
pitchers_rows = pitchers_sheet.nrows

relief_pitchers = []
starting_pitchers = []

fbase = []
sbase = []
tbase = []
ss = []
lf = []
cf = []
rf = []
c = []


def clean(x):
    x = str(x)
    new_str = ''
    for j in x:
        if j != '*' and j != 'w':
            new_str += j

    if new_str == '':
        new_str = 0

    return float(new_str)


def pitcher_type(x):
    pitcher_type_ = 's'
    for j in range(len(x)):
        if x[j:j+2] == 'R(':
            if int(x[j+2]) > 0:
                pitcher_type_ = 'r'
    return pitcher_type_


def get_position(x):
    pos = ''
    for j in x:
        if j == '-':
            return pos
        pos += j

    return pos


def get_steal(x):
    if x == 'E':
        return 0
    if x == 'D':
        return 0.2
    if x == 'C':
        return 0.4
    if x == 'B':
        return 0.6
    if x == 'A':
        return 0.8
    if x == 'AA':
        return 1

    return 0


def get_fielding(x):
    for j in range(len(x)):
        if x[j] == '-':
            return (5 - int(x[j+1])) / 4

    return 0


def sort(arr):
    for m in range(len(arr)):
        for l in range(len(arr) - 1):
            if arr[l][1] < arr[l+1][1]:
                arr[l][1], arr[l+1][1] = arr[l+1][1], arr[l][1]
                arr[l][0], arr[l + 1][0] = arr[l + 1][0], arr[l][0]
    return arr


# hitting algorithm
for i in range(1, hitters_rows):
    if hitters_sheet.cell_value(i, 1) != 'M' and hitters_sheet.cell_value(i, 1) != 'X':
        if hitters_sheet.cell_value(i, 4) >= 230:

            name = hitters_sheet.cell_value(i, 2)

            position = get_position(hitters_sheet.cell_value(i, 36))

            lh_pct_tb = (clean(hitters_sheet.cell_value(i, 9)) + 2*clean(hitters_sheet.cell_value(i, 11)) +
                         clean(hitters_sheet.cell_value(i, 6))) / 108
            lh_pct_outs = (108 - clean(hitters_sheet.cell_value(i, 8)) + clean(hitters_sheet.cell_value(i, 13)) +
                           hitters_sheet.cell_value(i, 6)) / 108

            rh_pct_tb = (clean(hitters_sheet.cell_value(i, 18)) + 2 * clean(hitters_sheet.cell_value(i, 20)) +
                         clean(hitters_sheet.cell_value(i, 15))) / 108
            rh_pct_outs = (108 - clean(hitters_sheet.cell_value(i, 17)) + clean(hitters_sheet.cell_value(i, 22)) +
                           hitters_sheet.cell_value(i, 15)) / 108

            runs_outs_ratio = (2*rh_pct_tb/rh_pct_outs + lh_pct_tb/lh_pct_outs) / 3

            stealing = get_steal(hitters_sheet.cell_value(i, 24))

            fielding = get_fielding(hitters_sheet.cell_value(i, 36))

            value = 0.6*runs_outs_ratio + 0.3*fielding + 0.1*stealing

            if position == '1b':
                fbase.append([name, value])
            if position == '2b':
                sbase.append([name, value])
            if position == '3b':
                tbase.append([name, value])
            if position == 'ss':
                ss.append([name, value])
            if position == 'lf':
                lf.append([name, value])
            if position == 'cf':
                cf.append([name, value])
            if position == 'rf':
                rf.append([name, value])
            if position == 'c':
                c.append([name, value])


# pitching algorithm
for i in range(1, pitchers_rows):
    if pitchers_sheet.cell_value(i, 1) != 'M' and pitchers_sheet.cell_value(i, 1) != 'X':
        pt = pitcher_type(pitchers_sheet.cell_value(i, 21))
        min_ip = 125
        if pt == 'r':
            min_ip = 40
        if pitchers_sheet.cell_value(i, 3) >= min_ip:

            name = pitchers_sheet.cell_value(i, 2)

            lh_pct_tb = (clean(pitchers_sheet.cell_value(i, 8)) + 2*clean(pitchers_sheet.cell_value(i, 10)) +
                         clean(pitchers_sheet.cell_value(i, 5))) / 108
            lh_pct_outs = (108 - clean(pitchers_sheet.cell_value(i, 7)) + clean(pitchers_sheet.cell_value(i, 11)) +
                           pitchers_sheet.cell_value(i, 4)) / 108

            rh_pct_tb = (clean(pitchers_sheet.cell_value(i, 16)) + 2 * clean(pitchers_sheet.cell_value(i, 18)) +
                         clean(pitchers_sheet.cell_value(i, 13))) / 108
            rh_pct_outs = (108 - clean(pitchers_sheet.cell_value(i, 15)) + clean(pitchers_sheet.cell_value(i, 19)) +
                           pitchers_sheet.cell_value(i, 12)) / 108

            # lower is better
            value = 1 - (lh_pct_tb/lh_pct_outs + rh_pct_tb/rh_pct_outs) / 2

            if value > 0:
                if pt == 'r':
                    relief_pitchers.append([name, value])
                elif pt == 's':
                    starting_pitchers.append([name, value])

relief_pitchers = sort(relief_pitchers)
starting_pitchers = sort(starting_pitchers)

fbase = sort(fbase)
sbase = sort(sbase)
tbase = sort(tbase)
ss = sort(ss)
lf = sort(lf)
cf = sort(cf)
rf = sort(rf)
c = sort(c)

for player in relief_pitchers:
    print(player[0], format(player[1], '1.3f'))
