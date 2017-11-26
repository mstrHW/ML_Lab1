import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split

class Lab1(object):

    def __init__(self, data):
        self.X = data.data
        self.Y = data.target
        self.feature_names = data.feature_names

    def plot_features(self):
        colors = ['r', 'b']

        feature_names = self.feature_names[:2]
        plt.scatter(self.X[:, 0], self.X[:, 1], c=colors)

        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])

        plt.show()

    def n_pca(self, n_components):
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(self.X)
        return X_reduced

    def get_covariance(self, X):
        X_cov = np.cov(X.T)
        return X_cov

    def get_orto(self, X):
        result = X.T.dot(X)
        return result

    def plot_pca(self, X_reduced):
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=self.Y,
                   cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_title("First three PCA directions")
        ax.set_xlabel("1st eigenvector")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd eigenvector")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd eigenvector")
        ax.w_zaxis.set_ticklabels([])
        plt.show()

    def k_neighbors(self, X, Y, k):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, Y_train)
        print('Train_set : n_neighbors : {}, accuracy : {:.8f}'.format(k, neigh.score(X_train, Y_train)))
        print('Test_set : n_neighbors : {}, accuracy : {:.8f}'.format(k, neigh.score(X_test, Y_test)))

    def k_neighbors_n_fold(self,  X, Y, k, cross_val_n=10, Print=False):
        neigh = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(neigh, X, Y, cv=cross_val_n)
        mean = scores.mean()
        if Print:
            print('Cross_val : {}'.format(mean))
        return mean

    def k_neighbors_plot(self, ks, accuracy, n):
        plt.scatter(ks, accuracy)

        plt.xlabel('K_neighbors, n_fold : {}'.format(n))
        plt.ylabel('Accuracy')

        plt.show()
