import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from skopt.acquisition import _gaussian_acquisition

def f(x):
    total = 0
    for exp, coef in enumerate(coefs):
        total += coef * (x ** exp)
    return total

if __name__ == '__main__':
    '''define true function'''
    coefs=[6, -2.5, -2.4, -.1, .2, .03]
    xs = np.linspace(-5., 3., 100)
    ys = f(xs)

    x_obs = np.array([-4, -1.5, 0, 1.5, 2.5, 2.7])
    y_obs = f(x_obs)
    x_s = np.linspace(-5, 3, 80)

    i = 0
    cov_amplitude = ConstantKernel(1.0)
    other_kernel = Matern()

    print(x_obs, y_obs)
    est = GaussianProcessRegressor(kernel=cov_amplitude*other_kernel)
    b_s = x_s
    s = []
    while i < len(x_s):
        x_obs = x_obs.reshape(len(x_obs), 1)
        x_s = x_s.reshape(len(x_s), 1)
        est.fit(x_obs, y_obs)
        values = _gaussian_acquisition(X=x_s, model=est)
        next_x = b_s[np.random.randint(0, 80)]
        s.append(next_x)
        x_obs = np.append(x_obs, np.array(next_x))
        y_obs = np.append(y_obs, f(next_x))
        i = i + 1

    import matplotlib.pyplot as plt
    y_mean, y_std = est.predict(np.array(s).reshape(len(s), 1), return_std=True)
    y_upper = y_mean + y_std
    y_lower = y_mean - y_std
    l1, l2, l3 = plt.plot(xs, ys, 'r-', s, y_mean, 'b-', s, y_upper, 'g-')
    plt.legend(handles=[l1, l2, l3], labels=['True Function', 'Mean', 'UPPER'])
    plt.show()