from sklearn import datasets
import numpy as np
from lab1 import Lab1


def task1(lab, n_pca, DEMO=False):

    # Построить диаграмму рассеяния для двух произвольно взятых признаков.
    lab.plot_features()

    #Рассчитать матрицу ковариации исходного набора данных (X) и для исходного набора, спроецированного на главные компоненты (X_reduced).
    X = lab.X
    X_cov = lab.get_covariance(X)

    X_reduced = lab.n_pca(n_pca)
    X_reduced_cov = lab.get_covariance(X_reduced)


    #Проверить, что главные компоненты ортогональны.
    orto = lab.get_orto(X_reduced) # Здесь есть вопрос
    #print(X_reduced_cov)


    #Сравнить собственные значения матрицы ковариации X со значениями дисперсии главных компонент.
    eigvalsX = np.linalg.eigvals(X_cov)
    eigvalsX_red = np.array([X_reduced_cov[i][i] for i in range(len(X_reduced_cov))])

    if (DEMO == True):
        print('Cобственные значения  матрицы ковариации : {}'.format(eigvalsX))
        print('Значениями дисперсии главных компонент : {}'.format(eigvalsX_red))


    #Рассчитать total variation (след матрицы ковариации) для X и X_reduced. Показать, что данный параметр не меняется при проецировании на главные компоненты.
    traceX = np.trace(X)
    traceX_red = np.trace(X_reduced)

    if (DEMO == True):
        print('Cлед матрицы ковариации для X : {}'.format(traceX))
        print('Cлед матрицы ковариации для X_reduced : {}'.format(traceX_red))

    #Построить графики % объясненной дисперсии: а) для исходных признаков, б) для главных компонент
    lab.plot_EV(eigvalsX, title='Для исходных признаков')
    lab.plot_EV(eigvalsX_red, title='Для главных компонент')

if __name__ == '__main__':
    data = datasets.load_breast_cancer()
    lab = Lab1(data)
    #[task1(n_pca) for n_pca in range(2, 15)]
    task1(lab, 10)



