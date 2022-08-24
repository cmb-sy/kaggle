import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def polynomial_regression(train_X, train_Y, X_star1, lamda):
    N, D = train_X.shape 
    matrix = np.eye(D, D)
    tmp = np.linalg.pinv(train_X.T @ train_X + lamda*matrix) @ (train_X.T @ train_Y)
    pridict = X_star1 @ tmp
    return pridict

if __name__ == '__main__':
    num = 200
    degree = 7
    x_min = -np.pi * 2
    x_max = np.pi * 2
    np.random.seed(0)
    lamda = 0
    
    x = np.linspace(x_min, x_max, num) 
    epsilon = np.random.normal(loc=0, scale=7e-1, size=len(x)) #ガウシアンノイズ
    y = 2* np.sin(x) + epsilon 
    train_X = np.concatenate((np.ones((num, 1)), x[:, None]), axis=1)

    for i in range(degree):
        tmp_x = x ** (i+1)
        train_X = np.concatenate((train_X, tmp_x[:, None]), axis=1)

    train_Y = y
    

    star_num = 500
    X_star =  np.linspace(x_min, x_max, star_num)
    
    X_star1 = np.concatenate((np.ones((star_num, 1)), X_star[:, None]), axis=1)
    for i in range(degree):
        tmp_x1 = X_star ** (i+1)
        X_star1 = np.concatenate((X_star1, tmp_x1[:, None]), axis=1)

    pridict_Y = polynomial_regression(train_X, train_Y, X_star1, lamda)
    plt.scatter(train_X[:, 1], y, s=20, c="b")
    plt.plot(X_star1[:, 1], pridict_Y, c="r", lw=5)

    plt.show()
