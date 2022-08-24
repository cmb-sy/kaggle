# リッジのコード整地ver

# jax.tree_multimapを使えば、もっと綺麗に勾配法が書ける
# 分けてaとbとしているので行列ではない。なぜか行列でまとめてするとうまくいかない
# 多項式回帰は更新すべきパラメータが多いので、勾配法は向いていないように思う。
# 局所多項式回帰が一番精度良い気がする
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from jax import grad
from matplotlib.gridspec import GridSpec
import sys

def model(x, params):
    f_x = 0
    cnt = 0
    keys_sorted = sorted(params)
    for key in keys_sorted:
      tmp = x**cnt
      f_x += params[key]*tmp
      cnt+=1
    return f_x

@jax.jit
def obf_Linear_regression(train_X, train_Y, params):
    Y_hat = model(train_X, params)
    E = jnp.power(train_Y - Y_hat, 2).mean()
    return E

def fit(train_X, train_Y, params, eta, epoch):
    history = dict(error=np.empty((epoch)),
                   param=np.empty((epoch, 2, 1)))

    for t in tqdm(range(epoch)):
        E = lambda params: obf_Linear_regression(train_X, train_Y, params)
        dEda = grad(E, argnums=(0))(params)        
        #キーを順に取り出している
        for key in params.keys():
          params[key] -= eta * dEda[key]
          if jnp.isnan(params[key]).any():
              print("[ERROR] NaN detected from params")
              sys.exit(1)
        history['error'][t] = obf_Linear_regression(train_X, train_Y, params)
    return params, history

if __name__ == '__main__':
    num = 10
    x_min = -np.pi
    x_max = np.pi 
    x = np.linspace(x_min, x_max, num)[:, None]
    eta = 1e-9
    epoch = 100000
    degree = 10
    params = {}

    for i in range(degree):
      key = "a" + str(i) 
      value =  np.random.normal(loc=0, scale=1e-10, size=x.shape[1:])
      params[key] = value
    
    epsilon = np.random.normal(loc=0, scale=0, size=x.shape) #ガウシアンノイズ
    y = np.sin(x) + epsilon 

    train_X = jnp.array(x)
    train_Y = jnp.array(y)

    # 新規データと学習
    num_star = 500
    x_star = np.linspace(x_min, x_max, num_star)
    paramss, history = fit(train_X, train_Y, params, eta, epoch)
    Y_star = model(x_star, paramss)

    # 描画 
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(nrows=1, ncols=2, height_ratios=[1])
    axes =[fig.add_subplot(gs[0, 0]),
         fig.add_subplot(gs[0, 1])]

    axes[0].scatter(x, y, s=10, c="b")
    axes[0].plot(x_star, Y_star, c="r")

    t = np.arange(len(history["error"]))
    axes[1].plot(t, history["error"])

    plt.show()
