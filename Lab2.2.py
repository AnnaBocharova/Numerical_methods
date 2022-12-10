import random
import numpy as np

EPS = 1e-6
DAMPING = 0.15

def random_webgraph(n):
    result = np.zeros([n, n])

    for i in range(n):
        for j in range(n):
            result[i, j] = bool(random.getrandbits(1))

    return result


def get_a_matrix(matrix): 
    n = len(matrix) 
    a_matrix = np.zeros([n, n])
    count = np.zeros([n])
    for j in range(n):
        for i in range(n): 
            if(matrix[i][j]==1): 
                count[j]+=1

    for j in range(n):
       for i in range(n): 
           if(matrix[i][j]==1): 
               matrix[i][j]=matrix[i][j]/count[j]
           if(count[j]==0): 
               matrix[i][j]=1/n
    return matrix


def power_method(matrix_a, eps):
    n=len(matrix_a)
    result = np.ones(n)

    while True:
        prev_result = result
        result = matrix_a @ result

        if np.linalg.norm(result - prev_result, 1) <= eps:
            break

    return result

def get_google_matrix(matrix, damping):
    n = len(matrix)
    matrix_a = get_a_matrix(matrix)
    matrix_b = np.ones((n, n)) / n
    return (1 - damping) * matrix_a + damping * matrix_b

print("WebGraph:")
matrix = random_webgraph(4)
print(matrix)
print("A_matrix:")
print(get_a_matrix(matrix))
print("Google matrix")
matrix_m = get_google_matrix(matrix, DAMPING)
print(matrix)
print("PageRank:")
print(power_method(matrix_m, EPS))


