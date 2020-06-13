from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
from sklearn import linear_model, svm
import warnings
from tqdm import tqdm


if __name__ == '__main__':

    """
    Partie 1
    """
    # trainx, trainy = gen_arti(nbex=1000, data_type=0, epsilon=1)
    # testx, testy = gen_arti(nbex=1000, data_type=0, epsilon=1)
    #
    # perceptron = Lineaire(hinge, hinge_g, max_iter=1000, eps=0.01)
    # perceptron.fit(trainx, trainy, modstep=False, min_iter=1, mode='mini-batch')
    # print("Erreur (lineaire) : train {0}, test {1}".format(
    #     perceptron.score(trainx, trainy), perceptron.score(testx, testy)
    # ))
    # plt.figure()
    # plot_frontiere(trainx, perceptron.predict, 200)
    # plot_data(trainx, trainy)
    #
    # perceptron_sklearn = linear_model.Perceptron(max_iter=1000, tol=None)
    # perceptron_sklearn.fit(trainx, trainy)
    # print("Erreur (sklearn) : train {0}, test {1}".format(
    #     perceptron_sklearn.score(trainx, trainy), perceptron_sklearn.score(testx, testy)
    # ))
    # plt.figure()
    # plot_frontiere(trainx, perceptron_sklearn.predict, 200)
    # plot_data(trainx, trainy)
    #
    # plt.show()

    """
    Partie 2
    """
    # figsize = (15, 7.5)
    # # On recupère et normalise les données
    # (trainx, trainy), (testx, testy) = mnist.load_data()
    # trainx = trainx.reshape(trainx.shape[0], -1)
    # testx = testx.reshape(testx.shape[0], -1)
    # trainx = trainx / np.max(trainx)
    # testx = testx / np.max(testx)
    #
    # min_iter, max_iter = 10, 1000
    # eps = 0.01
    # mode = 'batch'
    # modstep = False
    # mbnum = 5
    # use_tqdm = True
    # warnings.filterwarnings("ignore")
    # traincpt, testcpt = 8000, 2000
    # kernel = 'rbf'
    # # On met les données sous une forme plus convenable à utiliser
    # # datax, datay = np.vstack((trainx, testx)), np.concatenate((trainy, testy))
    # train = np.hstack((trainx, trainy.reshape(-1, 1)))
    # test = np.hstack((testx, testy.reshape(-1, 1)))
    # # data = np.hstack((datax, datay.reshape(-1, 1)))
    #
    # # Exemple de l'image dans la base de données
    # # plt.imshow(trainx[1].reshape(28, 28), interpolation='nearest', cmap='gray')
    #
    # # 6 vs les autres
    # # Préparation de données
    # train6not6, test6not6 = train[:traincpt], test[:testcpt]
    # train6not6[:, -1] = np.where(train6not6[:, -1] == 6, 1, -1)
    # test6not6[:, -1] = np.where(test6not6[:, -1] == 6, 1, -1)
    #
    # perceptron6not6 = Lineaire(hinge, hinge_g, max_iter=max_iter, eps=eps, use_tqdm=use_tqdm)
    # perceptron6not6.fit(
    #     train6not6[:, :-1], train6not6[:, -1], test6not6[:, :-1], test6not6[:, -1], modstep=modstep, min_iter=min_iter,
    #     mode=mode, mbnum=mbnum
    # )
    #
    # # Matrice de poids obtenue pour 6 vs les autres par perceptron simple
    # plt.figure(figsize=figsize)
    # plt.title('6 vs les autres, perceptron simple')
    # plt.subplot(121)
    # plt.imshow(perceptron6not6.w[:-1].reshape(28, 28), interpolation='nearest')
    # plt.colorbar()
    #
    # # Erreur en apprentissage et celle en test pour 6 vs les autres
    # # plt.figure()
    # # plt.title('6 vs les autres, perceptron simple')
    # plt.subplot(122)
    # plt.grid(True)
    # plt.plot(
    #     np.arange(len(perceptron6not6.losses['train'])), np.array(perceptron6not6.losses['train']), 'b-',
    #     label='Apprentissage'
    # )
    # plt.plot(
    #     np.arange(len(perceptron6not6.losses['test'])), np.array(perceptron6not6.losses['test']), 'r-',
    #     label='Test'
    # )
    # plt.xlabel('Époque')
    # plt.ylabel('Erreur')
    # plt.legend()
    #
    # print("Score de bonne classification (6 vs les autres) pour perceptron simple : train {0}, test {1}".format(
    #     perceptron6not6.score(train6not6[:, :-1], train6not6[:, -1]),
    #     perceptron6not6.score(test6not6[:, :-1], test6not6[:, -1])
    # ))
    #
    # reg_perceptron6not6 = Lineaire(
    #     reg_hinge, reg_hinge_g, max_iter=max_iter, eps=eps, use_tqdm=use_tqdm, bias=0.0, alpha=1.0
    # )
    # reg_perceptron6not6.fit(
    #     train6not6[:, :-1], train6not6[:, -1], test6not6[:, :-1], test6not6[:, -1], modstep=modstep, min_iter=min_iter,
    #     mode=mode, mbnum=mbnum
    # )
    #
    # # Matrice de poids obtenue pour 6 vs les autres par perceptron regularisé
    # plt.figure(figsize=figsize)
    # plt.title('6 vs les autres, perceptron regularisé')
    # plt.subplot(121)
    # plt.imshow(reg_perceptron6not6.w[:-1].reshape(28, 28), interpolation='nearest')
    # plt.colorbar()
    #
    # # Erreur en apprentissage et celle en test pour 6 vs les autres
    # # plt.figure()
    # # plt.title('6 vs les autres, perceptron regularisé')
    # plt.subplot(122)
    # plt.grid(True)
    # plt.plot(
    #     np.arange(len(reg_perceptron6not6.losses['train'])), np.array(reg_perceptron6not6.losses['train']), 'b-',
    #     label='Apprentissage'
    # )
    # plt.plot(
    #     np.arange(len(reg_perceptron6not6.losses['test'])), np.array(reg_perceptron6not6.losses['test']), 'r-',
    #     label='Test'
    # )
    # plt.xlabel('Époque')
    # plt.ylabel('Erreur')
    # plt.legend()
    #
    # print("Score de bonne classification (6 vs les autres) pour perceptron regularisé : train {0}, test {1}".format(
    #     reg_perceptron6not6.score(train6not6[:, :-1], train6not6[:, -1]),
    #     reg_perceptron6not6.score(test6not6[:, :-1], test6not6[:, -1])
    # ))
    #
    # svc = svm.SVC(max_iter=max_iter, kernel=kernel)
    # svc.fit(train6not6[:, :-1], train6not6[:, -1])
    # print("Score de bonne classification (6 vs les autres) pour svm (svc) : train {0}, test {1}".format(
    #     svc.score(train6not6[:, :-1], train6not6[:, -1]),
    #     svc.score(test6not6[:, :-1], test6not6[:, -1])
    # ))
    #
    # if kernel == 'linear':
    #     # Matrice de poids obtenue pour 6 vs les autres par svm (svc linéaire)
    #     plt.figure()
    #     plt.title('6 vs les autres, svm')
    #     plt.imshow(svc.coef_.reshape(28, 28), interpolation='nearest')
    #     plt.colorbar()
    #
    # plt.show()

    """
    Partie 3
    """
    figsize = (15, 7.5)
    # # On teste svm pour un noyau choisi
    kernel = 'rbf'
    max_iter = 1000
    trainx, trainy = gen_arti(nbex=1000, data_type=1, epsilon=0.4)
    testx, testy = gen_arti(nbex=1000, data_type=1, epsilon=0.4)

    # On entraîne notre modèle pour les données générées
    svc = svm.SVC(probability=True, kernel=kernel, max_iter=max_iter)
    svc.fit(trainx, trainy)
    print("Score de bonne classification (svm) : train {0}, test {1}".format(
        svc.score(trainx, trainy), svc.score(testx, testy)
    ))
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.title('Frontières')
    plot_frontiere_proba(trainx, lambda x: svc.predict_proba(x)[:, 0], 200)
    plot_data(trainx, trainy)

    # plt.figure()
    plt.subplot(122)
    plt.grid(True, alpha=0.5)
    plot_data(trainx[svc.support_], trainy[svc.support_])
    plt.title('Vecteurs supports')

    plt.show()

    # Grid search se trouve dans le fichier grid_search.py

    """
    Partie 4
    """
    # # On recupère et normalise les données
    # (trainx, trainy), (testx, testy) = mnist.load_data()
    # trainx = trainx.reshape(trainx.shape[0], -1)
    # testx = testx.reshape(testx.shape[0], -1)
    # trainx = trainx / np.max(trainx)
    # testx = testx / np.max(testx)
    #
    # max_iter = 1000
    # traincpt, testcpt = 1600, 400
    # kernel = 'rbf'
    # img_pred_num = 5
    # # On met les données sous une forme plus convenable à utiliser
    # # datax, datay = np.vstack((trainx, testx)), np.concatenate((trainy, testy))
    # train = np.hstack((trainx, trainy.reshape(-1, 1)))
    # test = np.hstack((testx, testy.reshape(-1, 1)))
    # train, test = train[:traincpt], test[:testcpt]
    # data = np.vstack((train, test))
    #
    # svc_ovr = svm.SVC(max_iter=max_iter, kernel=kernel)
    # svc_ovr.fit(train[:, :-1], train[:, -1])
    # print("Score de bonne classification (one-vs-rest) : train {0}, test {1}".format(
    #     svc_ovr.score(train[:, :-1], train[:, -1]),
    #     svc_ovr.score(test[:, :-1], test[:, -1])
    # ))
    #
    # # Quelques examples des prédictions pour one-vs-rest
    # img_inds = np.random.choice(np.arange(data.shape[0]), img_pred_num, replace=False)
    # for i in img_inds:
    #     plt.figure()
    #     plt.title("One-vs-rest. Prédiction : %d. Vrai label : %d" % (
    #         svc_ovr.predict(data[i, :-1].reshape(1, -1)), data[i, -1]
    #     ))
    #     plt.imshow(data[i, :-1].reshape(28, 28), interpolation='nearest', cmap='gray')
    #
    # svc_ovo = svm.SVC(max_iter=max_iter, kernel=kernel, decision_function_shape='ovo')
    # svc_ovo.fit(train[:, :-1], train[:, -1])
    # print("Score de bonne classification (one-vs-one) : train {0}, test {1}".format(
    #     svc_ovo.score(train[:, :-1], train[:, -1]),
    #     svc_ovo.score(test[:, :-1], test[:, -1])
    # ))
    #
    # # Quelques examples des prédictions pour one-vs-one
    # img_inds = np.random.choice(np.arange(data.shape[0]), img_pred_num, replace=False)
    # for i in img_inds:
    #     plt.figure()
    #     plt.title("One-vs-one. Prédiction : %d. Vrai label : %d" % (
    #         svc_ovo.predict(data[i, :-1].reshape(1, -1)), data[i, -1]
    #     ))
    #     plt.imshow(data[i, :-1].reshape(28, 28), interpolation='nearest', cmap='gray')
    #
    # plt.show()

