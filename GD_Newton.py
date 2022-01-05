# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:35:58 2021

@author: Abolfazl
"""

# -----------************* Question 1 *****************--------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#---------------- Define Gradient Descent Function ----------------------------------

def gradient_descent(max_iterations,threshold,w0,
                     cost_func,gradient_func,
                     learning_rate):
    
    w = w0
    w_history = w0
    f_history = cost_func(w)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 0.1
    
    while  i < max_iterations and diff > threshold:
        delta_w = (learning_rate)*(gradient_func(w))
        w = w - delta_w
        
        # store the history of w and f
        w_history = np.vstack((w_history,w))
        f_history = np.vstack((f_history,cost_func(w)))
        
        # update iteration number and diff between successive values
        # of objective function
        i+=1
        diff = np.absolute(f_history[-1]-f_history[-2])
        print(f"Iteration {i}: Cost {cost_func(w)}, Weight \
        {w}")
    #For 2-D Energy-Iteration Plot:
    plt.plot(f_history, marker='o', color='blue',markerfacecolor='red')
    plt.xlabel("Iteration", size = 14)
    plt.ylabel("Energy (cost)", size = 14)
    plt.show()
    #For 3-D trajectory plot:
    ax = plt.axes(projection ='3d')
    zz = f_history
    z = np.array(zz.flatten())
    xx = w_history[ :,0]
    yy = w_history[ :,1]
    weightsplot = np.array([xx.flatten(), yy.flatten()])
    x = weightsplot[0]
    y = weightsplot[1]
    ax.plot3D(x, y, z, 'green', marker='o' ,markerfacecolor='yellow')
    ax.set_title('3D line plot')
    plt.xlabel(" W1", size = 14)
    plt.ylabel(" W2", size = 14)
    plt.show()
    #return w_history,f_history

#------------ Energy (Cost) Function --------------------
def f(w):
    return (- (np.log(1-w[0]-w[1]) + np.log(w[0]) + np.log(w[1])))

#-------------- Gradient (g) Function ---------------
def grad(ww):
    g = []
    g1= ((1/(1-ww[0]-ww[1]) - (1/ww[0])))
    g.append(g1)
    
    g2= ((1/(1-ww[0]-ww[1]) - (1/ww[1])))
    g.append(g2)
    
    g = np.array(g)
    return g 

#-----------  Using Gradient Descent ----------
w0 = [0.2, 0.15]
w_init = np.array(w0)
learning_rate = 0.001
gradient_descent(1000, 0.001 ,w_init, f,grad,learning_rate)


#----------------------------- Define Newton's Function--------------------------------

def newton_hessian(max_iterations,threshold,w0h,
                     cost_func,gradient_func, hessian_func,
                     learning_rate):
    
    w = w0h
    w_history1 = w0h
    f_history1 = cost_func(w)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 0.1
    
    while  i < max_iterations and diff > threshold:
        delta_w = (learning_rate)*(np.matmul(hessian_func(w), gradient_func(w)))
        w = w - delta_w
        
        # store the history of w and f
        w_history1 = np.vstack((w_history1,w))
        f_history1 = np.vstack((f_history1,cost_func(w)))
        
        # update iteration number and diff between successive values
        # of objective function
        i+=1
        diff = np.absolute(f_history1[-1]-f_history1[-2])
        print(f"Iteration {i}: Cost {cost_func(w)}, Weight \
        {w}")
    #For 2-D Energy-Iteration Plot:
    plt.plot(f_history1, marker='o', color='blue',markerfacecolor='red')
    plt.xlabel("Iteration", size = 14)
    plt.ylabel("Energy (cost)", size = 14)
    plt.show()
    #For 3-D trajectory plot:
    ax = plt.axes(projection ='3d')
    zz = f_history1
    z = np.array(zz.flatten())
    xx = w_history1[ :,0]
    yy = w_history1[ :,1]
    weightsplot = np.array([xx.flatten(), yy.flatten()])
    x = weightsplot[0]
    y = weightsplot[1]
    ax.plot3D(x, y, z, 'green', marker='o',markerfacecolor='yellow')
    ax.set_title('3D line plot')
    plt.xlabel(" W1", size = 14)
    plt.ylabel(" W2", size = 14)
    plt.show()
    #return w_history1,f_history1


# ------------------------Define Hessian Function ------------------------------
def hessian(ww):
    h = []
    h1= ((1/((1-ww[0]-ww[1])**2)) + (1/((ww[0])**2)) ,   (1/((1-ww[0]-ww[1])**2)))
    h.append(h1)
    
    h2 = ((1/((1-ww[0]-ww[1])**2)) , (1/((1-ww[0]-ww[1])**2)) + (1/((ww[1])**2)))
    h.append(h2)
    
    h = np.array(h)
    return h 

# ------------------ Using Newton's method -------------
w01 = [0.2, 0.15]
w_init1 = np.array(w01)
learning_rate1 = 0.001
newton_hessian(1000, 0.001 ,w_init1, f, grad, hessian,learning_rate1)

# ------------------    ********* End of Question 1 *******************************    -------------
