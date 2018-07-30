import tensorflow as tf

coefs = [6, -2.5, -2.4, -.1, .2, .03]


def f(x):
    total = 0
    for exp, coef in enumerate(coefs):
        total += coef * (x ** exp)
    return total

