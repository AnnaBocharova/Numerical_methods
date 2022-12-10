import numpy as np 
a=-1
b=3
EPS=1e-3

def test1(x):
  return x**2-4

def test2(x):
  return 5*x**3-2*x**2*np.sin(x)-2/5

def d_fun(f, x):
  h = 1e-10
  return (f(x+h)-f(x))/h 

def dichotomy_method(f,a, b, eps):
  if f(a)*f(b)>0 : 
    print("Can't be solved by dichotomy method")
    return 
    
  x=(a+b)/2
  while np.abs(b-a) >= eps: 
    x=(a+b)/2
    if np.sign(f(a)) == np.sign(f(x)):
      a=x
    else: 
      b=x
  return x 


def relaxation_method(f,a, b, eps): 
  N=5000
  points=np.linspace(a,b) #the array is evenly distributed within the given interval
  m1=np.min(np.abs(d_fun(f, points)))
  M1=np.max(np.abs(d_fun(f, points)))

  t=2/(m1+M1) #optimal
  x=a
  prev=b
  while np.abs(x-prev)>eps: 
    sign=1
    prev=x
    if(d_fun(f, x)>0): 
      sign=-1
    x=prev+sign*t*f(prev)
  return x

def newton_method(f, a, b, eps): 
  if f(a)*f(b)>0 : 
    print("Can't be solved by dichotomy method")
    return 
  x=a
  prev=b
  while np.abs(x-prev)>eps: 
    prev=x 
    x=prev-f(x)/d_fun(f,x)
  return x

  


print ("Test func1: ")
print (dichotomy_method(test1, a, b, EPS))
print (relaxation_method(test1, a, b, EPS))
print (newton_method(test1, a, b, EPS))