import numpy as np 
import numpy.polynomial.polynomial as poly
import matplotlib
import matplotlib.pyplot as plt

def func(x):
    return 2*np.sin(x)+np.cos(3*x)

def lagrange_polinom (x_points, y_points) : 
    n = len(x_points); 
    res = poly.Polynomial([0])

    for i in range(n): 
        numerator = poly.Polynomial([1]) #1
        denominator = 1 

        for j in range(n): 
            if j!=i: 
                numerator*= poly.Polynomial([-x_points[j], 1]) #x - x_i 
                denominator *= x_points[i] - x_points[j]
        res+=y_points[i] * numerator / denominator
    return res 



def get_diff_table(x_points, y_points):
    n = len(x_points)
    table = np.zeros([n, n])
    table[:, 0] = np.copy(y_points)

    for i in range(1, n):
        for j in range(n - i):
            x_left = x_points[j]
            x_right = x_points[j + i]
            dx = x_right - x_left
            df = table[j + 1][i - 1] - table[j][i - 1]
            table[j][i] = df / dx

    return table


def newton_interpolation(x_points, y_points):
    result = poly.Polynomial([y_points[0]])
    n = len(x_points)
    diff_table = get_diff_table(x_points, y_points)
    polynom = poly.Polynomial([1])

    for i in range(1, n):
        polynom *= poly.Polynomial([-x_points[i - 1], 1]) 
        result += diff_table[0][i] * polynom

    return result


def get_h_i(x_points, y_points):
    intervals = len(x_points) - 1
    result = np.zeros(intervals)

    for i in range(intervals):
        result[i] = x_points[i + 1]- x_points[i]

    return result

def get_c_vector(x_points, y_points, arr_h) : 
    n = len(x_points)
    matrix_A = np.zeros([n-2,n-2])
    rhs_vector = np.zeros(n-2)
    c_vector = np.zeros(n)

    for i in range(n-2):
        matrix_A[i][i] = 2 * (arr_h[i] + arr_h[i + 1]) #C_i

        if i != 0:
            matrix_A[i][i - 1] = arr_h[i] #A_i

        if i != n - 3:
            matrix_A[i][i + 1] = arr_h[i + 1] #B_i

        rhs_vector[i] = 6 * ((y_points[i + 2] - y_points[i + 1]) / arr_h[i + 1] - (y_points[i + 1] - y_points[i]) / arr_h[i])


    c_vector[1:n - 1] = np.linalg.solve(matrix_A, rhs_vector) 
    return c_vector



def spline_interpolation(x_points, y_points):
    n = len(x_points)
    intervals = n - 1
    spline = np.empty([intervals], poly.Polynomial)
    arr_h = get_h_i(x_points, y_points)
    c_vector = get_c_vector(x_points, y_points, arr_h)   

    for i in range(1, intervals + 1):
        a_i = y_points[i]
        c_i = c_vector[i]
        d_i = (c_vector[i] - c_vector[i - 1]) / arr_h[i - 1]
        b_i = arr_h[i - 1] * c_vector[i] / 2 - arr_h[i - 1] ** 2 * d_i / 6 + (y_points[i] - y_points[i - 1]) / arr_h[i - 1]

        polynom = poly.Polynomial([-x_points[i], 1])
        s_i = poly.Polynomial([a_i])
        s_i += b_i * polynom
        s_i += (c_i / 2) * polynom ** 2
        s_i += (d_i / 6) * polynom ** 3

        spline[i - 1] = s_i

    return [spline, x_points]


def get_points_for_spline_plot(spline, x_points):
    n = len(x_points)
    result = np.zeros(n)
    number_spline = 0

    for i in range(n):
        x_i = x_points[i]

        if x_i > spline[1][number_spline + 1]:
            number_spline += 1

        result[i] = poly.polyval(x_i, spline[0][number_spline].coef)

    return result


x_start = -15
x_end = 15
n = 1000
n_points = 15

values = np.linspace(x_start, x_end, n)
x_points = np.linspace(x_start, x_end, n_points)
y_points = func(x_points)

lagrange = lagrange_polinom(x_points, y_points)
newton = newton_interpolation(x_points, y_points)
cyb_spline = spline_interpolation(x_points, y_points)

plt.plot(x_points, y_points, 'ko')
plt.plot(values, func(values), 'black', label='Function')
plt.plot(values, lagrange(values), 'red', label='Lagrange')
#plt.plot(values, newton(values), 'blue', label='Newton')
#plt.plot(values, get_points_for_spline_plot(cyb_spline, values), 'green', label='Spline')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()