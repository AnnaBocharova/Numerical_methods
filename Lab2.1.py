import numpy as np 
import random 

EPS=1e-5
UP=20
DOWN=1

def generate_matrix(n, type_matrix):
  matrix=[[random.randint(DOWN, UP) for i in range(n)] for j in range(n)]
  if(type_matrix=="Hilbert"): 
    for i in range(n): 
      for j in range(n):
        matrix[i][j]=1/(i+j-1)
  else:
    for i in range(n): 
      matrix[i][i]=np.sum(np.abs(matrix[i]))+1        
  return matrix 

def print_matrix(matrix): 
  for line in matrix:
    print ('  '.join(map(str, line)))

def generate_vector(n):
 return [random.randint(DOWN, UP) for i in range(n)]
  
def get_max_mistake(vctr): 
  return np.max(np.abs(vctr))
  
def jacobi_method(matrix, b, eps): 
  n=len(matrix)
  result=np.ones(n)

  while True: 
    prev=np.array(result)
    for i in range(n): 
      sum_coeff=0
      for j in range(n): 
        if i!=j:  
          sum_coeff+=matrix[i][j]*prev[j]
      result[i]=(b[i]-sum_coeff)/matrix[i][i]
    if get_max_mistake(result-prev)<eps: 
        break

  return result 

def seidel_method(matrix, b, eps): 
  n=len(matrix)
  result=np.ones(n) 

  while True: 
    prev=np.array(result)
    for i in range(n): 
      sum_coeff=0
      for j in range(n): 
        if j<i: 
          sum_coeff+=matrix[i][j]*result[j]
        elif j>i: 
          sum_coeff+=matrix[i][j]*prev[j]
      result[i]=(b[i]-sum_coeff)/matrix[i][i]
    
    if get_max_mistake(result-prev)<eps: 
      break 
     
  return result 

def find_leading_element(matrix, idx_column, idx_element):
  n=len(matrix)
  result=idx_element
  for i in range(idx_element+1, n): 
    if(np.abs(matrix[i][idx_column])>np.abs(matrix[result][idx_column])): 
      result=i
  return result

def get_p_matrix(n, k, l):
    result = np.identity(n)
    result[[k, l]] = result[[l, k]]
    return result

def get_m_matrix(current_matrix, k): 
  n = len(current_matrix)
  result = np.identity(n)
  result[k][k] = 1 / current_matrix[k][k]
  for i in range(k + 1, n):
    result[i][k] = -current_matrix[i][k] / current_matrix[k][k]

  return result

def gauss_method(matrix, b): 
  n=len(matrix)
  result=np.zeros(n)
  current_matrix = np.copy(matrix) 
  vector_b = np.copy(b)

  for i in range(n): 
    main_index=find_leading_element(current_matrix, i, i)
    p_matrix=get_p_matrix(n, i, main_index)
    current_matrix=p_matrix @ current_matrix
    m_matrix=get_m_matrix(current_matrix, i)
    current_matrix=m_matrix @ current_matrix 
    vector_b=m_matrix @ (p_matrix @ vector_b)

  for i in range(n - 1, -1, -1):
    result[i] = vector_b[i]
    for j in range(0, i):
      vector_b[j] -= current_matrix[j][i] * result[i]

  return result 


def menu():
  while True:
    print("1 --> Generate an ordinary matrix:")
    print("2 --> Generate a Hilbert matrix: ")
    print("0 --> Complete the work")
    cmd = input("Select item: ")
    
    if cmd == "1" or cmd=="2":
      if(cmd=="2") : cmd="Hilbert"
      n=int(input("Please, enter the dimension of the matrix: "))
      matrix=generate_matrix(n, cmd)
      b=generate_vector(n)
      print("----------Matrix----------")
      print_matrix(matrix)
      print("----------Vector b----------")
      print(b)
      print("----------Gauss----------")
      print(gauss_method(matrix, b))
      print("----------Jacobi----------")
      print(jacobi_method(matrix, b, EPS))
      print("----------Seidel----------")
      print(seidel_method(matrix, b, EPS))    
  
    elif cmd == "0":
      break
    else:
     print("You entered an invalid value, please try again:")

menu()