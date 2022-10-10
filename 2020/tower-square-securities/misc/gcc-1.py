def one():
    cases = int(input())
    for h in range(cases):
        r,c = input().split()
        r,c = int(r), int(c)
        initials = []
        for g in range(r):
            initial = input().split()
            for i in initial:
                initials.append(i)
        print(initials)
        

def two():
    cases = int(input())
    ans = []
    for i in range(cases):
        n = int(input())
        for i in range(n-1,0,-1):
            j = n - i
            if i + j == n:
                i, j = str(i), str(j)
                if "4" not in i and "4" not in j:
                    ans.append((i,j))
                    break
    for i in ans:
        print(i)




def three():
    tot = 0
    translate = {0:"",1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10: "ten",11:"eleven",12:"twelve",13:"thirteen",14:"fourteen",15:"fifteen",16:"sixteen",17:"seventeen",18:"eighteen",19:"nineteen",20:"twenty"}
    for i in range(1,10001):
        if i <= 20:
            tot += len(translate[i])
        elif i <= 30:
            i = str(i)
            end = translate[int(i[1])]
            tot += len(end)
            tot += len("thirty")
        elif i < 100:
            i = str(i)
            start = translate[int(i[0])]
            if start == "five":
                start = "fif"
            elif start == "four":
                start = "for"
            start += "ty"
            tot += len(start)
            tot += len(translate[int(i[1])])
        elif i <= 999:
            i = str(i)
            hundreds = translate[int(i[0])]
            hundreds += "hundredand"
            i = list(i)
            i.remove(i[0])
            i = int("".join(i))
            if i <= 20:
                tot += len(translate[i])
            elif i <= 30:
                i = str(i)
                end = translate[int(i[1])]
                tot += len(end)
                tot += len("thirty")
            elif i < 100:
                i = str(i)
                start = translate[int(i[0])]
                if start == "five":
                    start = "fif"
                elif start == "four":
                    start = "for"
                start += "ty"
                tot += len(start)
                tot += len(translate[int(i[1])])
            tot += len(hundreds)
        else:
            tot += len("onethousand")
                                 
            
    print(tot)





def four():
    amounts = []
    cases = int(input())
    for i in range(cases):
        amount = 0
        r,k,n = input().split()
        r,k,n = int(r), int(k), int(n)
        orig = k
        groups = input().split()
        groups = list(map(int,groups))
        end = []
        for x in range(r):
            for z in end:
                groups.remove(z)
                groups.append(z)
            end.clear()
            for y in groups:
                if k - y >= 0:
                    k -= y
                    amount += y
                    end.append(y)
                else:
                    break
            k = orig
        amounts.append(amount)

    for i, item in enumerate(amounts):
        print(f"Case #{i+1}: {item}")


# node.next
# node.val
n = ListNode(3)
list1 = []

def search(l, node):

    list1.append(node.val)

    if not node.next == None:
        search(l, node.next)


search(list1, n)