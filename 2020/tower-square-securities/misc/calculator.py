sigfigs = False
simple_adding = True


def add(a, b):
    if simple_adding:
        return a + b
    else:
        # get the length of num a
        len_a = 0
        if '.' in str(a):
            num, dec = str(a).split(sep='.')
            len_a = len(dec)

        # split b into values before and after decimal point, to be added separately
        str_b = str(b)
        if '.' in str_b:
            num, dec = str_b.split(sep='.')
        else:
            num = str_b
            dec = ''

        # add integer part of number
        for i in range(abs(int(num))):
            if b > 0:
                a += 1
            else:
                a -= 1

        # add decimal part of number
        increment = '0.'
        for i in dec:
            for j in range(int(i)):
                if b > 0:
                    a += float(increment + '1')
                else:
                    a -= float(increment + '1')
            increment += '0'

        # round value according to operation length to account for computer error
        if len(dec) > len_a:
            if sigfigs:
                a = round(a, len_a)
            else:
                a = round(a, len(dec))
        else:
            if sigfigs:
                a = round(a, len(dec))
            else:
                a = round(a, len_a)

        return a


def mult(a, b):

    carry = 0
    lines = []
    line_index = 0
    # nested loop through a and b to simulate long multiplication by hand
    for i in str(b)[::-1]:
        if i == '-' or i == '.':
            continue
        lines.append([])
        for j in str(a)[::-1]:
            if j == '-' or j == '.':
                continue
            total = 0
            for k in range(int(i)):
                total = add(total, int(j))
            total = add(total, carry)

            # get carry value and save value in line
            if len(str(total)) >= 2:
                carry = int(str(total)[0])
                lines[line_index].insert(0, str(total)[1])
            else:
                carry = 0
                lines[line_index].insert(0, str(total))

        if carry > 0:
            lines[line_index].insert(0, str(carry))
            carry = 0

        # put zeros on end
        for j in range(line_index):
            lines[line_index].append('0')

        line_index = add(line_index, 1)

    # combine lines
    result = 0
    for line in lines:
        this_line = ''
        for i in line:
            this_line += i
        result = add(result, int(this_line))

    # account for negativity
    if (str(a)[0] == '-' or str(b)[0] == '-') and not (str(a)[0] == '-' and str(b)[0] == '-'):
        result = int('-' + str(result))

    # place decimal point in result
    a_decimals = len(str(a)) - str(a).index('.') - 1 if '.' in str(a) else 0
    b_decimals = len(str(b)) - str(b).index('.') - 1 if '.' in str(b) else 0

    place = len(str(result)) - a_decimals - b_decimals
    if place < len(str(result)):
        result = float(str(result)[:place] + '.' + str(result)[place:])

    return result


def div(a, b):
    index = 1
    val = str(a).replace('-', '').replace('.', '')
    total = [0]

    a_decimals = len(str(a)) - str(a).index('.') - 1 if '.' in str(a) else 0
    b_decimals = len(str(b)) - str(b).index('.') - 1 if '.' in str(b) else 0
    place = a_decimals - b_decimals

    break_point = 5
    point = 0
    while int(val) > 0 and len(total) < 20 and point < break_point:
        if int(val[:index]) >= b:
            val = str(add(mult(b, -1), int(val[:index]))) + val[index:]
            total[index-1] = add(total[index-1], 1)
        elif 0 < int(val) < b:
            val += '0'
            point = add(point, 1)
            total.append(0)
        else:
            total.append(0)
            index += 1

        print(val)
    return total


print(div(5743, 23))
