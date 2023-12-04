import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sympy as sp
import numpy as np

from scipy.optimize import minimize

import random


def stock_uni(X_lower, Y_upper, t_date, N_points):
    X_lower = X_lower + 1
    Y_upper = Y_upper + 1
    # assume origin price is 1
    money_array = [1]* N_points
    price = 1

    for _ in range(t_date):
        temp = price
        price *= random.uniform(X_lower, Y_upper)
        money_array = [x * (1-i/N_points) + x * (i/N_points) / temp * price for i, x in enumerate(money_array)]
    
    #print("price: ", price)
    return money_array

def stock_cal_uni(X_lower, Y_upper):
    # assume origin price is 1, 0 < X_lower < 1, Y_upper > 1
    f = sp.symbols('f')
    s = sp.symbols('s')

    expression = sp.ln(1 + f * s)*(1 /(Y_upper - X_lower))
    
    G_f = sp.integrate(expression, (s, X_lower, Y_upper))
    print(G_f)
    # Convert G_f to a lambda function for scipy.optimize.minimize
    G_f_lambda = sp.lambdify(f, -G_f)

    # Define the constraints
    constraints = (
        {'type': 'ineq', 'fun': lambda f: f},  # f > 0
        {'type': 'ineq', 'fun': lambda f: 1 - f}  # f < 1
    )

    # Initial guess
    initial_guess = [0.5]

    # Find the Maximizer
    result = minimize(G_f_lambda, initial_guess, constraints=constraints)

    # Print the Maximizer
    print("Maximizer:", result.x)

    # Convert G_f to a lambda function for scipy.optimize.minimize
    G_f_lambda = sp.lambdify(f, G_f)
    # Generate f values
    f_values = np.linspace(0.01, 0.99, 100)

    # Compute G_f values
    G_f_values = [G_f_lambda(f_val) for f_val in f_values]
    # Plot G_f
    plt.plot(f_values, G_f_values)
    maximizer = result.x[0]
    plt.scatter([maximizer], [G_f_lambda(maximizer)], color='red')
    plt.annotate(f'Maximizer: {maximizer:.3f}', (maximizer, G_f_lambda(maximizer)), textcoords="offset points", xytext=(-10,-10), ha='center')
    plt.xlabel('f')
    plt.ylabel('G(f)')
    plt.title('G(f) over f')
    plt.savefig('G(f) over f.png')
    plt.show()
    

    


'''

# uniform with single stock
returns = stock_uni(-0.8,1,1000,1000)
max_index = returns.index(max(returns))
plt.plot(returns)
plt.xlabel('invest ratio‰')
plt.ylabel('Return Value(log)')
plt.title('Return Value over different invest ratio/Uniform Distribution')
plt.yscale('log')
plt.figtext(0.25, 0.2, 'Max Return Value: ' + "{:.3e}".format(max(returns)) + ' at invest ratio: ' + str(max_index/1000))
plt.savefig('Return Value over different invest ratio_Uniform Distribution.png')
plt.show()



'''

# Initialize a list to store the max_index values
max_indices = []

# Perform Monte Carlo simulation
for _ in range(1000):
    returns = stock_uni(-0.8, 1, 1000, 1000)
    max_index = returns.index(max(returns))
    max_indices.append(max_index)

# Plot the distribution of max_index
plt.hist(max_indices, bins=range(min(max_indices), max(max_indices) + 1), alpha=0.7, edgecolor='black')
plt.xlabel('Max Index‰')
plt.ylabel('Frequency')
plt.title('Distribution of Max Index')
plt.savefig('Distribution of Max Index.png')
plt.show()
# stock_cal_uni(-0.8, 1)