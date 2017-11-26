from sklearn import datasets
from lab1 import Lab1

data = datasets.load_breast_cancer()

lab = Lab1(data)

def task3(n_pca):

    X_reduced = lab.n_pca(n_pca)

    lab.k_neighbors(X_reduced, lab.Y, 7)

    lab.k_neighbors_n_fold(X_reduced, lab.Y, 7, 10)

    ns = [2, 5, 8, 10]
    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    nfold_accuracies = [[lab.k_neighbors_n_fold(X_reduced, lab.Y, k, n) for k in ks] for n in ns]
    for nfold in range(len(nfold_accuracies)):
        accuracies = nfold_accuracies[nfold]
        n = ns[nfold]
        lab.k_neighbors_plot(ks, accuracies, n)

task3(15)