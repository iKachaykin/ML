import numpy as np
import RegressionClassifier as rc
import matplotlib.pyplot as plt
from USPS.ManageUSPS import load_usps, show_usps


if __name__ == '__main__':
    figsize=(15, 7.5)
    alpha = {'linear': 1.0, 'ridge': 1.0, 'lasso': 0.01}
    min_iter, max_iter = 10, 50
    eps = 0.001
    mode = 'sgd'
    modstep = False
    mbnum = 10
    loss = {'linear': 'Régression linéaire', 'ridge': 'Régression ridge', 'lasso': 'LASSO'}
    # On recupère les données
    trainx, trainy = load_usps('USPS/USPS_train.txt')
    testx, testy = load_usps('USPS/USPS_test.txt')
    # datax, datay = np.vstack((trainx, testx)), np.concatenate((trainy, testy))
    train = np.hstack((trainx, trainy.reshape(-1, 1)))
    test = np.hstack((testx, testy.reshape(-1, 1)))
    # data = np.hstack((datax, datay.reshape(-1, 1)))

    # Exemple d'une chiffre qui est stockée dans la base de données
    # show_usps(trainx[0])

    # 6 vs 9
    print('6 vs 9\n')
    train6, train9 = train[train[:, -1] == 6], train[train[:, -1] == 9]
    test6, test9 = test[test[:, -1] == 6], test[test[:, -1] == 9]
    train69, test69 = np.vstack((train6, train9)), np.vstack((test6, test9))

    # On remplace des valeurs par défaut par -1 et 1 pour adopter des données à notre classifieur binaire
    train69[:, -1] = np.where(train69[:, -1] == 6, 1, -1)
    test69[:, -1] = np.where(test69[:, -1] == 6, 1, -1)

    for l in loss.keys():
        regression69 = rc.RegressionClassifier(
            alpha=alpha[l], loss=l, max_iter=max_iter, eps=eps, use_tqdm=True, min_iter=min_iter, modefit=mode,
            modstep=modstep, mbnum=mbnum, ignore_warnings=True
        )
        regression69.fit(train69[:, :-1], train69[:, -1], test69[:, :-1], test69[:, -1])

        # Matrice de poids obtenue pour 6 vs 9
        # print(perceptron69.w[:-1].reshape(16, 16))
        plt.figure(figsize=figsize)
        plt.suptitle(loss[l])
        plt.subplot(121)
        plt.imshow(regression69.w[:-1].reshape(16, 16), interpolation='nearest')
        plt.colorbar()

        # Erreur en apprentissage et celle en test pour 6 vs 9
        # plt.figure()
        plt.subplot(122)
        plt.grid(True)
        plt.plot(
            np.arange(len(regression69.losses['train'])), np.array(regression69.losses['train']), 'b-',
            label='Apprentissage'
        )
        plt.plot(
            np.arange(len(regression69.losses['test'])), np.array(regression69.losses['test']), 'r-',
            label='Test'
        )
        plt.xlabel('Époque')
        plt.ylabel('Erreur')
        plt.legend()

        print("Score de bonne classification ({0}) : train {1}, test {2}".format(
            loss[l], regression69.score(train69[:, :-1], train69[:, -1]),
            regression69.score(test69[:, :-1], test69[:, -1])
        ))
        print("Valeur de l'évaluation (%s) : %.12f" % (
            loss[l], regression69.evaluate(train69[:, :-1], train69[:, -1], test69[:, :-1], test69[:, -1])
        ))
        print("Norme euclidienne de vecteur w : {0}".format(np.linalg.norm(regression69.w)))

    plt.show()
