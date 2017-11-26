from sklearn import datasets
import numpy as np
from lab1 import Lab1

data = datasets.load_breast_cancer()

lab = Lab1(data)

lab.plot_features()

def task1(n_pca):

    X = lab.X
    X_cov = lab.get_covariance(X)

    X_reduced = lab.n_pca(n_pca)
    X_reduced_cov = lab.get_covariance(X_reduced)

    orto = lab.get_orto(X_reduced) # Здесь есть вопрос

    self_values = np.linalg.eigvals(X_cov)
    vars = np.array([X_reduced_cov[i][i] for i in range(len(X_reduced_cov))])

    #print(self_values)
    #print(vars)

    trace1 = np.trace(X)
    trace2 = np.trace(X_reduced)

    #print(trace1)
    #print(trace2)

    #def erv(self_values, vars):
    #    return 1 - (sum(vars ** 2) / sum(self_values ** 2))

    #initial_vars = np.array([X_cov[i][i] for i in range(len(X_cov))])
    #print(erv(self_values, initial_vars))

[task1(n_pca) for n_pca in range(2, 15)]



