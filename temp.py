'''
import random

# Define the probabilities of events A and B
probability_A = 0.3
probability_B = 0.6

# Define the conditional probability of B given A
conditional_probability_B_given_A = 0.8

# Function to simulate events A and B
def simulate_events():
    # Simulate event A
    event_A = random.random() < probability_A

    # Simulate event B based on the occurrence of A
    if event_A:
        event_B = random.random() < conditional_probability_B_given_A
    else:
        event_B = random.random() < probability_B

    return event_A, event_B

# Simulate events multiple times
num_simulations = 1000
count_A_and_B = 0

for _ in range(num_simulations):
    event_A, event_B = simulate_events()

    # Check if both events A and B occurred
    if event_A and event_B:
        count_A_and_B += 1

# Calculate the simulated probability of A and B occurring together
simulated_probability_A_and_B = count_A_and_B / num_simulations

print(f"Simulated probability of A and B occurring together: {simulated_probability_A_and_B}")
'''
# I get how to compute kelly invest when A and B are independent and bio 
import sympy as sp

f1 = sp.symbols('f1')
f2 = sp.symbols('f2')
p1 = 0.55
p2 = 0.55
q1 = 1 - p1
q2 = 1 - p2
b1 = 1
b2 = 1
c1 = -0.5
c2 = -0.5

expression = p1 * p2 * sp.ln(1 + b1 * f1 + b2 * f2) + p1 * q2 * sp.ln(1 + b1 * f1 + c2 * f2) + \
             q1 * p2 * sp.ln(1 + c1 * f1 + b2 * f2) + q1 * q2 * sp.ln(1 + c1 * f1 + c2 * f2)

derivative_result = sp.diff(expression, f1)

print(derivative_result)

from scipy.optimize import minimize

# Define the function to maximize
def func_to_maximize(f):
    f1, f2 = f
    return -(p1 * p2 * sp.log(1 + b1 * f1 + b2 * f2) + p1 * q2 * sp.log(1 + b1 * f1 + c2 * f2) + \
             q1 * p2 * sp.log(1 + c1 * f1 + b2 * f2) + q1 * q2 * sp.log(1 + c1 * f1 + c2 * f2))

# Define the constraints
constraints = (
    {'type': 'ineq', 'fun': lambda f: f[0]},  # f1 > 0
    {'type': 'ineq', 'fun': lambda f: f[1]},  # f2 > 0
    {'type': 'ineq', 'fun': lambda f: 1 - f[0]},  # f1 < 1
    {'type': 'ineq', 'fun': lambda f: 1 - f[1]},  # f2 < 1

    #add constraint
    #{'type': 'ineq', 'fun': lambda f: 1 - f[0] - f[1]}  # f1 + f2 < 1
)

# Initial guess
initial_guess = [0.4, 0.4]

# Find the maximizer
result = minimize(func_to_maximize, initial_guess, constraints=constraints)

# Print the maximizer
print("Maximizer:", result.x)