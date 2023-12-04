import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sympy as sp
import numpy as np

from scipy.optimize import minimize

import random

def stock_bio_2(p_rise, b_win, a_loss, t_date, N_points):
    # Create the first row with 100 elements for each number from 1 to 100
    invest_1 = np.repeat(np.arange(0, N_points), N_points)
    #print(invest_1.shape[0])
    invest_2 = np.tile(np.arange(0, N_points), N_points)

    matrix = np.vstack((invest_1, invest_2))
    # Calculate the third row as the sum of the first and second rows
    money = np.repeat(1.0, N_points**2)
    #print(money.shape[0])

    for _ in range(t_date):
        pro_1 = random.random()
        pro_2 = random.random()
        if pro_1 < p_rise:
            if pro_2 < p_rise:
                money = money * (1 + b_win * (matrix[0, :] / N_points + matrix[1, :] / N_points))
            else:
                money = money * (1 + b_win * (matrix[0, :] / N_points) - a_loss * (matrix[1, :] / N_points))
        elif pro_2 < p_rise:
                money = money * (1 + b_win * (matrix[1, :] / N_points) - a_loss * (matrix[0, :] / N_points))
        else:
                money = money * (1 - a_loss * (matrix[0, :] / N_points + matrix[1, :] / N_points))
    
    matrix = np.vstack((matrix, money))

    # add constraint f1 + f2 < 1
    mask = np.sum(matrix[:2,:], axis=0) <= 100
    matrix = matrix[:, mask]

    return matrix

def stock_cal_bio_3(p1,p2,p3,p4, b1, b2, c1 ,c2):
    f1 = sp.symbols('f1')
    f2 = sp.symbols('f2')
    c1 = -c1
    c2 = -c2

    expression = p1 * sp.ln(1 + b1 * f1 + b2 * f2) + p2 * sp.ln(1 + b1 * f1 + c2 * f2) + \
                p3 * sp.ln(1 + c1 * f1 + b2 * f2) + p4 * sp.ln(1 + c1 * f1 + c2 * f2)

    derivative_result = sp.diff(expression, f1)

    print(derivative_result)

    from scipy.optimize import minimize

    # Define the function to maximize
    def func_to_maximize(f):
        f1, f2 = f
        return -(p1 * sp.ln(1 + b1 * f1 + b2 * f2) + p2 * sp.ln(1 + b1 * f1 + c2 * f2) + \
                 p3 * sp.ln(1 + c1 * f1 + b2 * f2) + p4 * sp.ln(1 + c1 * f1 + c2 * f2))

    # Define the constraints
    constraints = (
        {'type': 'ineq', 'fun': lambda f: f[0]},  # f1 > 0
        {'type': 'ineq', 'fun': lambda f: f[1]},  # f2 > 0
        {'type': 'ineq', 'fun': lambda f: 1 - f[0]},  # f1 < 1
        {'type': 'ineq', 'fun': lambda f: 1 - f[1]},  # f2 < 1

        #add constraint
        {'type': 'ineq', 'fun': lambda f: 1 - f[0] - f[1]}  # f1 + f2 < 1
    )

    # Initial guess
    initial_guess = [0.4, 0.4]

    # Find the maximizer
    result = minimize(func_to_maximize, initial_guess, constraints=constraints)

    # Print the maximizer
    print("Maximizer:", result.x)

    # Convert expression to a lambda function for plotting
    expression_lambda = sp.lambdify((f1, f2), expression)

    # Generate f1 and f2 values
    f1_values = np.linspace(0.01, 0.99, 1000)
    f2_values = np.linspace(0.01, 0.99, 1000)

    # Compute expression values
    expression_values = np.array([[expression_lambda(f1_val, f2_val) for f1_val in f1_values] for f2_val in f2_values])

    # Create a meshgrid for f1 and f2 values
    f1_values, f2_values = np.meshgrid(f1_values, f2_values)


    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a color map based on the condition x + y < 1
    colors = np.where(f1_values + f2_values > 1, 'r', 'b')

    # Plot the surface with the color map
    ax.plot_surface(f1_values, f2_values, expression_values, facecolors=colors)

    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('G(f1, f2)')
    plt.title('G(f1, f2) over f1 and f2')
    plte = plt.savefig('G(f1, f2) over f1 and f2.png')
    plt.show()

stock_cal_bio_3(0.45,0.1,0.1,0.35,1.0,1.0,0.5,0.5)

'''
matrix = stock_bio_2(0.55, 1, 0.5, 1000, 100)
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, and z values from your matrix
x_values = matrix[0, :]
y_values = matrix[1, :]
z_values = matrix[2, :]

z_values = np.log(z_values)
max_z_index = np.argmax(z_values)

# Plot the 3D scatter plot
ax.scatter(x_values, y_values, z_values)

# Add labels
ax.set_xlabel('f1 ratio')
ax.set_ylabel('f2 ratio')
ax.set_zlabel('Return Value(log)')


# Show the plot
plt.title('Return Value in two stocks')
plt.savefig('Return Value(log) over f1 and f2.png')
plt.show()
print("Point with maximum z-value:")
print("x =", x_values[max_z_index])
print("y =", y_values[max_z_index])
print("z =", z_values[max_z_index])
'''