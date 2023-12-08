# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 17:24:39 2022

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
get_ipython().run_line_magic('matplotlib', 'inline')
import random


# ## 2 -  Problem Statement
# 
# Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.
# - You would like to expand your business to cities that may give your restaurant higher profits.
# - The chain already has restaurants in various cities and you have data for profits and populations from the cities.
# - You also have data on cities that are candidates for a new restaurant. 
#     - For these cities, you have the city population.
#     
# Can you use the data to help you identify which cities may potentially give your business higher profits?
# 
# ## 3 - Dataset
# 
# You will start by loading the dataset for this task. 
# - The `load_data()` function shown below loads the data into variables `x_train` and `y_train`
#   - `x_train` is the population of a city
#   - `y_train` is the profit of a restaurant in that city. A negative value for profit indicates a loss.   
#   - Both `X_train` and `y_train` are numpy arrays.

# In[6]:


# load the dataset
x_train = [191, 255, 225, 160, 264, 211]
y_train = [8, 2, 4, 6, -2, 12]

x_train = np.dot(x_train, 0.1)
# #### View the variables
# Before starting on any task, it is useful to get more familiar with your dataset.  
# - A good place to start is to just print out each variable and see what it contains.
# 
# The code below prints the variable `x_train` and the type of the variable.

# In[53]:


# print x_train
print("Type of x_train:",type(x_train))
print("Elements of x_train are:\n", x_train) 


# `x_train` is a numpy array that contains decimal values that are all greater than zero.
# - These values represent the city population times 10,000
# - For example, 6.1101 means that the population for that city is 61,101
#   
# Now, let's print `y_train`

# In[54]:


# print y_train
print("Type of y_train:",type(y_train))
print("Elements of y_train are:\n", y_train)  

# In[56]:


# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Card on the top of the deck')
# Set the x-axis label
plt.xlabel('Card sum')
plt.show()

# In[10]:


# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    print(type(x))
    
    # You need to return this variable correctly
    total_cost = 0
    cost = 0
    m = len(x)
    
    ### START CODE HERE ###
    
    for i in range(m):
        f_wb = x[i] * w + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost
    ### END CODE HERE ### 

    return total_cost

# In[11]:


# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')
# In[16]:

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = len(x)
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = x[i] * w + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
        
    return dj_dw, dj_db
# In[17]:


# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

# In[19]:


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing


# Now let's run the gradient descent algorithm above to learn the parameters for our dataset.

# In[20]:


# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 10000000
alpha = 0.001

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)