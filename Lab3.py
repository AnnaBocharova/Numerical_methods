import math

import numpy as np
EPS = 1e-10

#System 1 
#f1 = x^2/y^2 - cos(y) - 2 = 0 
#f2 = x^2 + y^2 - 6 = 0


def first_func1(values): 
    return values[0]**2/values[1]**2 - math.cos(values[1]) - 2

def first_func2(values): 
    return values[0]**2 + values[1]**2 - 6

def first_func(values): 
    return np.array([first_func1(values), first_func2(values)])

def first_df1dx(values): 
    return 2*values[0]/values[1]**2 

def first_df1dy(values): 
    return math.sin(values[1])-2*values[0]**2/values[1]**3

def first_df2dx(values): 
    return 2*values[0]

def first_df2dy(values): 
    return 2*values[1]

def first_jacobi_matrix(values): 
    return np.array([
        [first_df1dx(values), first_df1dy(values)], 
        [first_df2dx(values), first_df2dy(values)]
        ])


#System 2
#f_1 = x_1^3 + x_2^2 + x_3^2 + ... + x_n^2 - 1^3 - 2^2 - 3^2 - ... - n^2 = 0
#f_2 = x_1^2 + x_2^3 + x_3^2 + ... + x_n^2 - 1^2 - 2^3 - 3^2 - ... - n^2 = 0
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
#f_n = x_1^2 + x_2^2 + x_3^2 + ... + x_n^3 - 1^2 - 2^2 - 3^2 - ... - n^3 = 0


def second_func(values): 
    n = values.shape[0]
    res = np.zeros(n)

    for i in range(n): 
       lhs = 0
       rhs = 0 

       for j in range(n): 
           lhs_add=values[j]**2
           rhs_add=(j+1)**2 
           if j == i : 
               lhs_add*=values[j]
               rhs_add*=(j+1)
           lhs+=lhs_add
           rhs+=rhs_add

       res[i] = lhs - rhs

    return res 

       
def second_dfi_dxj(values, i, j) :
    if i==j : return 3*values[j]**2 
    else : return 2*values[j]

def second_jacobi_matrix(values) : 
    n = len(values)
    res = np.empty([n, n]); 

    for i in range(n): 
        for j in range(n): 
            res[i, j] = second_dfi_dxj(values, i, j)

    return res 


#Methods 

def newton_method(x_0, eps, func, jacobi_matrix_func, is_modified) : 
    res = np.copy(x_0) 
    jacobi_matrix = jacobi_matrix_func(res)

    while True : 
        if is_modified==False:
            jacobi_matrix = jacobi_matrix_func(res)
        res_func = func(res)
        delta_x = np.linalg.solve(jacobi_matrix, res_func)
        res = res - delta_x 
        if(np.linalg.norm(delta_x, 1)) <= eps : 
            break

    return res 

def max_norm(matrix): 
    abs_min = abs(np.min(matrix))
    abs_max = abs(np.max(matrix))
    if abs_min > abs_max : return abs_min 
    else : return abs_max

def relaxation_method(x_0, eps, jacobi_matrix_func, func) : 
    res = np.copy(x_0)
    jacobi_matrix = jacobi_matrix_func(res)
    tau = (2/max_norm(jacobi_matrix))/10 
    
    while True : 
        prev_res = np.copy(res)
        res = res - func(res) * tau 

        if np.linalg.norm(res - prev_res, 1) <= eps : 
            break;

    return res

print ("-------------------------System 1----------------------------")
x_0 = np.ones(2)
print ("Newton method: ")
res = newton_method(x_0, EPS, first_func, first_jacobi_matrix, False)
invariance = np.linalg.norm(first_func(res), 1)
print(f"Result newton method system1: {res}")
print(f"Нев'язка : {invariance}")

print ("-------------------------------------------------------")
print("Modified newwton method")
res = newton_method(x_0, 0.01, first_func, first_jacobi_matrix, True)
invariance = np.linalg.norm(first_func(res), 1)
print(f"Result modified newton method system1: {res}")
print(f"Нев'язка : {invariance}")

print ("-------------------------------------------------------")
print("Relaxantion method: ")
res = relaxation_method(x_0, EPS, first_jacobi_matrix, first_func)
invariance = np.linalg.norm(first_func(res), 1)
print(f"Result relaxatiom method system1: {res}")
print(f"Нев'язка : {invariance}")

print ("-------------------------------------------------------")
print ("-------------------------------------------------------")
print ("-------------------------------------------------------")
print ("-------------------------System 2----------------------------")
#x_0 = np.ones(5) - переповнення метод модифікований і релаксації
x_0 = np.array([1.2, 2.2, 3.2, 4.2, 5.2])
print ("Newton method: ")
res = newton_method(x_0, EPS, second_func, second_jacobi_matrix, False)
invariance = np.linalg.norm(second_func(res), 1)
print(f"Result newton method system2: {res}")
print(f"Нев'язка : {invariance}")

print ("-------------------------------------------------------")
print("Modified newwton method")
res = newton_method(x_0, EPS, second_func, second_jacobi_matrix, True)
invariance = np.linalg.norm(second_func(res), 1)
print(f"Result modified newton method system2: {res}")
print(f"Нев'язка : {invariance}")

print ("-------------------------------------------------------")
print("Relaxantion method: ")
res = relaxation_method(x_0, EPS, second_jacobi_matrix, second_func)
invariance = np.linalg.norm(second_func(res), 1)
print(f"Result relaxatiom method system2: {res}")
print(f"Нев'язка : {invariance}")
