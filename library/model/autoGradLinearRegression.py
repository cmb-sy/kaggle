# jax.tree_multimapを使えば、もっと綺麗に勾配法が書ける
# 分けてaとbとしているので行列ではない。なぜか行列でまとめてするとうまくいかない
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from jax import grad
from matplotlib.gridspec import GridSpec

def model(x, param):
    return param['a0']*x + param['a1']

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
        params['a0'] -= eta * dEda['a0']
        params['a1'] -= eta * dEda['a1']
        history['error'][t] = obf_Linear_regression(train_X, train_Y, params)
    return params, history

if __name__ == '__main__':
    num = 200
    x_min = 0
    x_max = 10
    x = np.linspace(x_min, x_max, num)[:, None]
    eta = 1e-3
    epoch = 1000

    params = {
    'a0': jnp.zeros(x.shape[1:]), #列数の獲得(2, 3)にすると(1, 3)が得られる
    'a1': 0.
    }
    
    epsilon = np.random.normal(loc=0, scale=5e-1, size=x.shape) #ガウシアンノイズ
    y = x + epsilon 

    train_X = jnp.array(x)
    train_Y = jnp.array(y)

    # 新規データと予測
    num_star = 300
    x_star = np.linspace(x_min, x_max, num_star)
    params, history = fit(train_X, train_Y, params, eta, epoch)
    Y_star = model(x_star, params)

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(nrows=1, ncols=2, height_ratios=[1])
    axes =[fig.add_subplot(gs[0, 0]),
         fig.add_subplot(gs[0, 1])]

    axes[0].scatter(x, y, s=10, c="b")
    axes[0].plot(x_star, Y_star, c="r")

    t = np.arange(len(history["error"]))
    axes[1].plot(t, history["error"])

    plt.show()
