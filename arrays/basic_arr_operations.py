import math

# Add 2 arrays 
def add(a, b):
    res = []
    for e1, e2 in zip(a, b):
        res.append(e1+e2)
    return res

def mult_and_sum(a, b):
    res =  0
    for x,y in zip(a, b):
        res += x * y
    return res

if __name__ == '__main__':
    x = [1, 2, 3, 4]
    y = [1,1,1,1]
    arr_addition = add(x, y)
    # print(f"array_addition = {arr_addition}")
    arr_sum = mult_and_sum(x, y)
    # print(f"array_mult_and_sum = {arr_sum}")