import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sympy as sp
import numpy as np

from scipy.optimize import minimize
import random

def stock_bio(p_rise, b_win, a_loss, t_date, N_points):
    money_array = [1]* N_points
    number_rise = 0
    number_fall = 0

    for _ in range(t_date):
        if random.random() < p_rise:
            money_array = [x*(1 + b_win * (i/N_points))  for i, x in enumerate(money_array)]
            number_rise += 1
        else:
            money_array = [x*(1 - a_loss * (i/N_points))  for i, x in enumerate(money_array)]
            number_fall += 1
    
    print("number of rise: ", number_rise)
    print("number of fall: ", number_fall)
    
    return money_array


def stock_cal_bio(p_rise, b_win, a_loss):
    # assume origin price is 1, 0 < X_lower < 1, Y_upper > 1
    f = sp.symbols('f')
    q_fall = 1 - p_rise

    G_f = p_rise * sp.ln(1 + f * b_win) + q_fall * sp.ln(1 - f * a_loss )
    
    print(G_f)
    # Convert G_f to a lambda function for scipy.optimize.minimize
    G_f_lambda = sp.lambdify(f, -G_f)

    # Define the constraints
    constraints = (
        {'type': 'ineq', 'fun': lambda f: f},  # f > 0
        {'type': 'ineq', 'fun': lambda f: 1 - f}  # f < 1
    )

    initial_guess = [0.5]

    # Find the Maximizer
    result = minimize(G_f_lambda, initial_guess, constraints=constraints)

    # Print the Maximizer
    print("Maximizer:", result.x)

    G_f_lambda = sp.lambdify(f, G_f)
    # Generate f values
    f_values = np.linspace(0.01, 0.99, 100)

    G_f_values = [G_f_lambda(f_val) for f_val in f_values]
    # Plot G_f
    plt.plot(f_values, G_f_values)
    maximizer = result.x[0]
    plt.scatter([maximizer], [G_f_lambda(maximizer)], color='red')
    plt.annotate(f'Maximizer: {maximizer:.3f}', (maximizer, G_f_lambda(maximizer)), textcoords="offset points", xytext=(-10,-10), ha='center')
    plt.xlabel('f')
    plt.ylabel('G(f)')
    plt.title('G(f) over f')
    plt.show()

#stock_cal_bio(0.55, 1, 0.5)

# bio with single stock
returns = stock_bio(0.55, 1, 0.5, 1000, 1000)
max_index = returns.index(max(returns))
plt.plot(returns)
plt.xlabel('invest ratio')
plt.ylabel('Return Value')
plt.title('Return Value over different invest ratio')

plt.yscale('log')
plt.figtext(0.3, 0.2, 'Max Return Value: ' + "{:.3e}".format(max(returns)) + ' at invest ratio: ' + str(max_index/1000))
plt.savefig('bio_single_stock.png')
plt.show()
# it should be 65%
'''
# Initialize a list to store the max_indices
max_indices = []

# Run stock_bio 100 times
for _ in range(1000):
    returns = stock_bio(0.55, 1, 0.5, 1000, 1000)
    max_index = returns.index(max(returns))
    max_indices.append(max_index)

# Plot the distribution of max_index
plt.hist(max_indices, bins=range(min(max_indices), max(max_indices) + 1), alpha=0.7, edgecolor='black')
plt.xlabel('Max Index')
plt.ylabel('Frequency')
plt.title('Distribution of Max Index')
plt.savefig('bio_single_stock_hist.png')
plt.show()
'''