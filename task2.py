from sklearn import datasets
from lab1 import Lab1


def task2(lab):

    #Обучить классификатор на основе метода k ближайших соседей при фиксированном (произвольном) k.
    #Оценить качество классификации на тренировочной/тестовой выборке.
    lab.k_neighbors(lab.X, lab.Y, 7, Print=True)

    #Сделать 10-fold кросс-валидацию при фиксированном k
    lab.k_neighbors_n_fold(lab.X, lab.Y, 7, 10, Print=True)

    ns = [2, 5, 8, 10]
    ks = [i for i in range(1, 10)]

    #Построить графики зависимости точности на тренировочном/тестовом наборе от числа k (с дисперсией)
    nfold_accuracies = [[lab.k_neighbors_n_fold(lab.X, lab.Y, k, n) for k in ks] for n in ns]
    for nfold in range(len(nfold_accuracies)):
        accuracies = [i[0] for i in nfold_accuracies[nfold]]
        errors = [i[1] for i in nfold_accuracies[nfold]]
        n = ns[nfold]
        best_accuracy = max(accuracies)
        print('Best_accuracy : {} on k_neighbors : {}'.format(best_accuracy, accuracies.index(best_accuracy)))
        lab.k_neighbors_plot(ks, accuracies, errors, n)


if __name__ == '__main__':
    data = datasets.load_breast_cancer()
    lab = Lab1(data)

    task2(lab)

