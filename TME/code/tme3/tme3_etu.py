from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings


def mse(datax, datay, w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    datax = np.array(datax)
    if len(datax.shape) == 1:
        datax = np.array([datax])
    datay = np.array(datay)
    if len(datay.shape) == 0:
        datay = np.array([datay])
    return np.mean((datay - np.dot(datax, w)) ** 2)


def mse_g(datax, datay, w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    datax = np.array(datax)
    if len(datax.shape) == 1:
        datax = np.array([datax])
    datay = np.array(datay)
    if len(datay.shape) == 0:
        datay = np.array([datay])
    return -2 * np.mean((datay - np.dot(datax, w)).reshape(-1, 1) * datax, axis=0)


def hinge(datax, datay, w):
    """ retourn la moyenne de l'erreur hinge """
    datax = np.array(datax)
    if len(datax.shape) == 1:
        datax = np.array([datax])
    datay = np.array(datay)
    if len(datay.shape) == 0:
        datay = np.array([datay])
    return np.mean(np.maximum(0.0, -datay * np.dot(datax, w)))


def hinge_g(datax, datay, w):
    """ retourne le gradient moyen de l'erreur hinge """
    datax = np.array(datax)
    if len(datax.shape) == 1:
        datax = np.array([datax])
    datay = np.array(datay)
    if len(datay.shape) == 0:
        datay = np.array([datay])
    return np.mean(np.where(
        (-datay * np.dot(datax, w)).reshape(-1, 1) * np.ones_like(datax) > 0,
        -datay.reshape(-1, 1) * datax, 0.0),
        axis=0
    )


class Lineaire(object):
    def __init__(self, loss=hinge, loss_g=hinge_g, max_iter=1000, eps=0.01):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        self.w = None
        self.wlist = []
        self.losses = {'train': [], 'test': []}

    def fit(
            self, datax, datay, testx=None, testy=None, weps=1e-10, feps=1e-10, geps=1e-10, min_iter=10, modstep=False,
            mode='batch', mbnum=None
    ):
        """
        Wrapper de fit par des descents de gradient différents.

        :trainx: donnees de train
        :trainy: label de train
        :testx: donnees de test
        :testy: label de test
        :weps: tolérance pour w
        :feps: tolérance pour f
        :geps: tolérance pour g_f
        :min_iter: nombre minimal des itérations à faire
        :modstep: flag qui indique s'il faut mettre à jour un learning rate ou pas (selon la méthode de
        Barzilai–Borwein https://doi.org/10.1093/imanum/8.1.141) ; n'utiliser que si mode == 'batch'
        :mode: mode de descent de gradient à utiliser ; soit 'batch', soit 'sgd', soit 'mini-batch'
        :mbnum: nombre de mini-batches à créer
        """
        if modstep and mode != 'batch':
            warnings.warn('modstep was True but would not be used unless mode == "batch"')
        if mbnum is not None and mode != 'mini-batch':
            warnings.warn('mbnum was not None but would not be used unless mode == "mini-batch"')
        if mode == 'batch':
            self._fit_batch(datax, datay, testx, testy, weps, feps, geps, min_iter, modstep)
        elif mode == 'sgd':
            self._fit_sgd(datax, datay, testx, testy, weps, feps, geps, min_iter)
        elif mode == 'mini-batch':
            self._fit_mini_batch(
                datax, datay, testx, testy, weps, feps, geps, min_iter, mbnum if mbnum is not None else 2
            )
        else:
            raise ValueError('mode not understood!')

    def _fit_batch(
            self, datax, datay, testx=None, testy=None, weps=1e-10, feps=1e-10, geps=1e-10, min_iter=10, modstep=False
    ):
        """
        Descent de gradient par batch. On y minimise le gradient moyen.

        :trainx: donnees de train
        :trainy: label de train
        :testx: donnees de test
        :testy: label de test
        :weps: tolérance pour w
        :feps: tolérance pour f
        :geps: tolérance pour g_f
        :min_iter: nombre minimal des itérations à faire
        :modstep: flag qui indique s'il faut mettre à jour un learning rate ou pas (selon la méthode de
        Barzilai–Borwein https://doi.org/10.1093/imanum/8.1.141)
        """
        if weps is None:
            weps = self.eps
        if feps is None:
            feps = self.eps
        if geps is None:
            geps = self.eps
        if testx is not None and testy is not None:
            self.losses['test'] = []
            self.losses['train'] = []
            testx = np.hstack((testx, np.ones((len(testy), 1))))
            lstats = True
        else:
            lstats = False
        N = len(datay)
        datax = datax.reshape(N, -1)
        D = datax.shape[1]
        # On ajoute une colonne de 1 pour considérer un bias
        datax = np.hstack((datax, np.ones((N, 1))))
        self.w = np.random.rand(D+1)
        step = self.eps
        self.wlist = [self.w.copy()]
        for i in range(self.max_iter):
            wpred, fpred, g = self.w.copy(), self.loss(datax, datay, self.w), self.loss_g(datax, datay, self.w)
            if lstats:
                self.losses['train'].append(fpred)
                self.losses['test'].append(self.loss(testx, testy, self.w))
            self.w = self.w - step * g
            g, gpred = self.loss_g(datax, datay, self.w), g.copy()
            self.wlist.append(self.w.copy())
            if (i >= min_iter and np.linalg.norm(self.w - wpred) < weps and
                    np.abs(self.loss(datax, datay, self.w) - fpred) < feps and
                    (np.linalg.norm(g) < geps or np.linalg.norm(g - gpred) < geps)):
                print(i)
                break
            if modstep:
                step = (
                    np.abs(np.dot(self.w - wpred, g - gpred)) / np.linalg.norm(g - gpred) ** 2
                    if np.all(g != gpred)
                    else 0.0)
        if lstats:
            self.losses['train'].append(self.loss(datax, datay, self.w))
            self.losses['test'].append(self.loss(testx, testy, self.w))

    def _fit_sgd(
            self, datax, datay, testx=None, testy=None, weps=1e-10, feps=1e-10, geps=1e-10, min_iter=10):
        """
        Descent de gradient par sgd.

        :trainx: donnees de train
        :trainy: label de train
        :testx: donnees de test
        :testy: label de test
        :weps: tolérance pour w
        :feps: tolérance pour f
        :geps: tolérance pour g_f
        :min_iter: nombre minimal des itérations à faire
        """
        if weps is None:
            weps = self.eps
        if feps is None:
            feps = self.eps
        if geps is None:
            geps = self.eps
        if testx is not None and testy is not None:
            self.losses['test'] = []
            self.losses['train'] = []
            testx = np.hstack((testx, np.ones((len(testy), 1))))
            lstats = True
        else:
            lstats = False
        N = len(datay)
        datax = datax.reshape(N, -1)
        D = datax.shape[1]
        # On ajoute une colonne de 1 pour considérer un bias
        datax = np.hstack((datax, np.ones((N, 1))))
        self.w = np.random.rand(D+1)
        step = self.eps
        self.wlist = [self.w.copy()]
        for i in range(self.max_iter):
            wpred, fpred, gpred = self.w.copy(), self.loss(datax, datay, self.w), self.loss_g(datax, datay, self.w)
            if lstats:
                self.losses['train'].append(fpred)
                self.losses['test'].append(self.loss(testx, testy, self.w))
            data_shuffled = np.hstack((datax, datay.reshape(-1, 1)))
            np.random.shuffle(data_shuffled)
            datax_shuffled, datay_shuffled = data_shuffled[:, :-1], data_shuffled[:, -1]
            for xline, yline in zip(datax_shuffled, datay_shuffled):
                self.w = self.w - step * self.loss_g(xline, yline, self.w)
            self.wlist.append(self.w.copy())
            g = self.loss_g(datax, datay, self.w)
            if (i >= min_iter and np.linalg.norm(self.w - wpred) < weps and
                    np.abs(self.loss(datax, datay, self.w) - fpred) < feps and np.linalg.norm(g) < geps):
                print(i)
                break
        if lstats:
            self.losses['train'].append(self.loss(datax, datay, self.w))
            self.losses['test'].append(self.loss(testx, testy, self.w))

    def _fit_mini_batch(
            self, datax, datay, testx=None, testy=None, weps=1e-10, feps=1e-10, geps=1e-10, min_iter=10, mbnum=2):
        """
        Descent de gradient par mini-batch.

        :trainx: donnees de train
        :trainy: label de train
        :testx: donnees de test
        :testy: label de test
        :weps: tolérance pour w
        :feps: tolérance pour f
        :geps: tolérance pour g_f
        :min_iter: nombre minimal des itérations à faire
        :mbnum: nombre de mini-batches à créer
        """
        if weps is None:
            weps = self.eps
        if feps is None:
            feps = self.eps
        if geps is None:
            geps = self.eps
        if testx is not None and testy is not None:
            self.losses['test'] = []
            self.losses['train'] = []
            testx = np.hstack((testx, np.ones((len(testy), 1))))
            lstats = True
        else:
            lstats = False
        N = len(datay)
        datax = datax.reshape(N, -1)
        D = datax.shape[1]
        # On ajoute une colonne de 1 pour considérer un bias
        datax = np.hstack((datax, np.ones((N, 1))))
        self.w = np.random.rand(D+1)
        step = self.eps
        self.wlist = [self.w.copy()]
        for i in range(self.max_iter):
            wpred, fpred = self.w.copy(), self.loss(datax, datay, self.w)
            if lstats:
                self.losses['train'].append(fpred)
                self.losses['test'].append(self.loss(testx, testy, self.w))
            data_stacked = np.hstack((datax, datay.reshape(-1, 1)))
            np.random.shuffle(data_stacked)
            batches = [[] for _ in range(mbnum)]
            for dline in data_stacked:
                bind = np.random.randint(0, mbnum, 1)[0]
                batches[bind].append(dline)
            for bind in range(mbnum):
                batches[bind] = np.array(batches[bind])

            for batch in batches:
                self.w = self.w - step * self.loss_g(batch[:, :-1], batch[:, -1], self.w)

            self.wlist.append(self.w.copy())
            g = self.loss_g(datax, datay, self.w)
            if (i >= min_iter and np.linalg.norm(self.w - wpred) < weps and
                    np.abs(self.loss(datax, datay, self.w) - fpred) < feps and np.linalg.norm(g) < geps):
                print(i)
                break
        if lstats:
            self.losses['train'].append(self.loss(datax, datay, self.w))
            self.losses['test'].append(self.loss(testx, testy, self.w))

    def predict(self, datax):
        """
        On prédit des labels pour les données passées

        :param datax: données pour lesquelles il faut prédire des labels
        :return yprn: prédiction des labels
        """
        if len(datax.shape) == 1:
            datax = datax.reshape(1, -1)
        N = datax.shape[0]
        datax = np.hstack((datax, np.ones((N, 1))))
        return np.sign(np.dot(datax, self.w))

    def score(self, datax, datay):
        """
        Calcul du pourcentage de bonne classification

        :param datax: données pour lesquelles on prédit des labels
        :param datay: vrais labels de données passées
        :return score: un nombre normalisée sur [0, 1] de bonnes classifications
        """
        return np.mean(np.where(datay - self.predict(datax) == 0, 1, 0))


def load_usps(fn):
    with open(fn, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:, 1:], tmp[:, 0].astype(int)


def show_usps(data):
    plt.imshow(data.reshape((16, 16)), interpolation="nearest", cmap="gray")


def plot_error(datax, datay, f, step=10, wlist=None):
    grid, x1list, x2list = make_grid(xmin=-4, xmax=4, ymin=-4, ymax=4)
    plt.contourf(x1list, x2list, np.array([f(datax, datay, w) for w in grid]).reshape(x1list.shape), 25)
    plt.colorbar()
    if wlist is not None:
        wlist_show = [wlist[0]]
        for w in wlist[1:-1]:
            if np.linalg.norm(wlist_show[-1][:2] - w[:2]) > 0.1:
                wlist_show.append(w)
        wlist_show.append(wlist[-1])
        plt.plot(np.array(wlist_show)[:, 0], np.array(wlist_show)[:, 1], 'k-')
        plt.plot(np.array(wlist_show)[0, 0], np.array(wlist_show)[0, 1], 'bo', label='Initiale')
        plt.plot(np.array(wlist_show)[1:-1, 0], np.array(wlist_show)[1:-1, 1], 'ko', label='Intérmidiaires')
        plt.plot(np.array(wlist_show)[-1, 0], np.array(wlist_show)[-1, 1], 'ro', label='Finale')
        plt.legend()


def projection_quad(x):
    x = np.array(x)
    if len(x.shape) == 1:
        x = np.array([x])
    return np.array([np.ones(x.shape[0]), x[:, 0], x[:, 1], x[:, 0] ** 2, x[:, 1] ** 2, x[:, 0] * x[:, 1]]).T


def projection_gauss(x, rect=None, grid_size=None):
    x = np.array(x)
    if len(x.shape) == 1:
        x = np.array([x])
    if grid_size is None:
        grid_size = tuple([10 for _ in range(x.shape[1])])
    if rect is None:
        rect = np.zeros((x.shape[1], 2))
        rect[:, 0] = np.floor(np.min(x, axis=0))
        rect[:, 1] = np.ceil(np.max(x, axis=0))
    ranges = np.array([
        np.linspace(ximin, ximax, xisize)
        for ximin, ximax, xisize in zip(rect[:, 0], rect[:, 1], grid_size)
    ])
    rrlist = list(np.meshgrid(*tuple(ranges)))
    grid = np.c_[tuple([rr.ravel() for rr in rrlist])]
    proj = np.zeros((x.shape[0], grid.shape[0]))
    for i in range(x.shape[0]):
        proj[i] = (2*np.pi) ** (-x[i].shape[-1]/2) * np.exp(-0.5*np.linalg.norm(x[i] - grid, axis=1)**2)
    return proj


if __name__ == "__main__":
    """
    Partie 1
    """
    # # plt.ion()
    # figsize=(15, 7.5)
    # trainx, trainy = gen_arti(nbex=1000, data_type=0, epsilon=1)
    # testx, testy = gen_arti(nbex=1000, data_type=0, epsilon=1)

    # perceptron = Lineaire(hinge, hinge_g, max_iter=1000, eps=0.01)
    # perceptron.fit(trainx, trainy, modstep=False, min_iter=1, mode='mini-batch')
    # print("Erreur (perceptron) : train {0}, test {1}".format(
    #     perceptron.score(trainx, trainy), perceptron.score(testx, testy)
    # ))
    # plt.figure(figsize=figsize)
    # plt.subplot(121)
    # plot_frontiere(trainx, perceptron.predict, 200)
    # plot_data(trainx, trainy)

    # # plt.figure()
    # plt.subplot(122)
    # plot_error(trainx, trainy, hinge, wlist=perceptron.wlist)

    # regression = Lineaire(mse, mse_g, max_iter=1000, eps=0.01)
    # regression.fit(trainx, trainy, modstep=False, min_iter=1, mode='mini-batch')
    # print("Erreur (regression) : train {0}, test {1}".format(
    #     regression.score(trainx, trainy), regression.score(testx, testy)
    # ))
    # plt.figure(figsize=figsize)
    # plt.subplot(121)
    # plot_frontiere(trainx, regression.predict, 200)
    # plot_data(trainx, trainy)

    # # plt.figure()
    # plt.subplot(122)
    # plot_error(trainx, trainy, mse, wlist=regression.wlist)

    # plt.show()

    """
    Partie 2
    """
    # figsize=(15, 7.5)
    # min_iter, max_iter = 10, 1000
    # eps = 0.01
    # mode = 'mini-batch'
    # modstep = False
    # mbnum = 5
    # warnings.filterwarnings("ignore")
    # # On recupère les données
    # trainx, trainy = load_usps('USPS/USPS_train.txt')
    # testx, testy = load_usps('USPS/USPS_test.txt')
    # # datax, datay = np.vstack((trainx, testx)), np.concatenate((trainy, testy))
    # train = np.hstack((trainx, trainy.reshape(-1, 1)))
    # test = np.hstack((testx, testy.reshape(-1, 1)))
    # # data = np.hstack((datax, datay.reshape(-1, 1)))
    #
    # # Exemple d'une chiffre qui est stockée dans la base de données
    # # show_usps(trainx[0])
    #
    # # 6 vs 9
    # train6, train9 = train[train[:, -1] == 6], train[train[:, -1] == 9]
    # test6, test9 = test[test[:, -1] == 6], test[test[:, -1] == 9]
    # train69, test69 = np.vstack((train6, train9)), np.vstack((test6, test9))
    #
    # # On remplace des valeurs par défaut par -1 et 1 pour adopter des données à notre classifieur binaire
    # train69[:, -1] = np.where(train69[:, -1] == 6, 1, -1)
    # test69[:, -1] = np.where(test69[:, -1] == 6, 1, -1)
    #
    # perceptron69 = Lineaire(hinge, hinge_g, max_iter=max_iter, eps=eps)
    # perceptron69.fit(
    #     train69[:, :-1], train69[:, -1], test69[:, :-1], test69[:, -1], modstep=modstep, min_iter=min_iter, mode=mode,
    #     mbnum=mbnum
    # )
    #
    # # Matrice de poids obtenue pour 6 vs 9
    # # print(perceptron69.w[:-1].reshape(16, 16))
    # plt.figure(figsize=figsize)
    # plt.subplot(121)
    # plt.title('6 vs 9')
    # plt.imshow(perceptron69.w[:-1].reshape(16, 16), interpolation='nearest')
    # plt.colorbar()
    #
    # # Erreur en apprentissage et celle en test pour 6 vs 9
    # # plt.figure()
    # plt.subplot(122)
    # plt.title('6 vs 9')
    # plt.grid(True)
    # plt.plot(
    #     np.arange(len(perceptron69.losses['train'])), np.array(perceptron69.losses['train']), 'b-',
    #     label='Apprentissage'
    # )
    # plt.plot(
    #     np.arange(len(perceptron69.losses['test'])), np.array(perceptron69.losses['test']), 'r-',
    #     label='Test'
    # )
    # plt.xlabel('Époque')
    # plt.ylabel('Erreur')
    # plt.legend()
    #
    # # Valeur de l'erreur pour l'échantillon de train
    # # print(hinge(np.hstack(
    # #     (train69[:, :-1], np.ones(train69.shape[0]).reshape(-1, 1))), train69[:, -1], perceptron69.w)
    # # )
    #
    # print("Score de bonne classification (6 vs 9) : train {0}, test {1}".format(
    #     perceptron69.score(train69[:, :-1], train69[:, -1]), perceptron69.score(test69[:, :-1], test69[:, -1])
    # ))
    #
    # # 6 vs les autres
    # # Préparation de données
    # train6not6, test6not6 = train.copy(), test.copy()
    # train6not6[:, -1] = np.where(train6not6[:, -1] == 6, 1, -1)
    # test6not6[:, -1] = np.where(test6not6[:, -1] == 6, 1, -1)
    #
    # perceptron6not6 = Lineaire(hinge, hinge_g, max_iter=max_iter, eps=eps)
    # perceptron6not6.fit(
    #     train6not6[:, :-1], train6not6[:, -1], test6not6[:, :-1], test6not6[:, -1], modstep=modstep, min_iter=min_iter,
    #     mode=mode, mbnum=mbnum
    # )
    #
    # # Matrice de poids obtenue pour 6 vs les autres
    # plt.figure(figsize=figsize)
    # plt.subplot(121)
    # plt.title('6 vs les autres')
    # plt.imshow(perceptron6not6.w[:-1].reshape(16, 16), interpolation='nearest')
    # plt.colorbar()
    #
    # # Erreur en apprentissage et celle en test pour 6 vs les autres
    # # plt.figure()
    # plt.subplot(122)
    # plt.title('6 vs les autres')
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
    # print("Score de bonne classification (6 vs les autres) : train {0}, test {1}".format(
    #     perceptron6not6.score(train6not6[:, :-1], train6not6[:, -1]),
    #     perceptron6not6.score(test6not6[:, :-1], test6not6[:, -1])
    # ))
    #
    # plt.show()

    """
    Partie 3
    """
    figsize = (15, 7.5)
    trainx, trainy = gen_arti(nbex=1000, data_type=1, epsilon=0.1)
    testx, testy = gen_arti(nbex=1000, data_type=1, epsilon=0.1)

    perceptron = Lineaire(hinge, hinge_g, max_iter=1000, eps=0.01)
    perceptron.fit(trainx, trainy, modstep=False, min_iter=1, mode='mini-batch')
    print("Erreur (perceptron) : train {0}, test {1}".format(
        perceptron.score(trainx, trainy), perceptron.score(testx, testy)
    ))
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plot_frontiere(trainx, perceptron.predict, 200)
    plot_data(trainx, trainy)

    regression = Lineaire(mse, mse_g, max_iter=1000, eps=0.01)
    regression.fit(trainx, trainy, modstep=False, min_iter=1, mode='mini-batch')
    print("Erreur (regression) : train {0}, test {1}".format(
        regression.score(trainx, trainy), regression.score(testx, testy)
    ))
    plt.subplot(122)
    plot_frontiere(trainx, regression.predict, 200)
    plot_data(trainx, trainy)

    # On fait une projection polynomiale
    ptrainx, ptestx = projection_quad(trainx), projection_quad(testx)

    # On entraîne notre modèle pour les données projetées
    perceptronp = Lineaire(hinge, hinge_g, max_iter=1000, eps=0.01)
    perceptronp.fit(ptrainx, trainy, modstep=False, min_iter=10, mode='mini-batch')
    print("Score de bonne classification (polynomial) : train {0}, test {1}".format(
        perceptronp.score(ptrainx, trainy), perceptronp.score(ptestx, testy)
    ))
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plot_frontiere(trainx, lambda x: perceptronp.predict(projection_quad(x)), 200)
    plot_data(trainx, trainy)

    # La même chose mais avec une projection gaussienne
    grid_size = (15, 15)
    gtrainx, gtestx = projection_gauss(trainx, grid_size=grid_size), projection_gauss(testx, grid_size=grid_size)

    perceptrong = Lineaire(hinge, hinge_g, max_iter=1000, eps=0.01)
    perceptrong.fit(gtrainx, trainy, modstep=False, min_iter=10, mode='mini-batch')
    print("Score de bonne classification (gauss) : train {0}, test {1}".format(
        perceptrong.score(gtrainx, trainy), perceptrong.score(gtestx, testy)
    ))
    # plt.figure()
    plt.subplot(122)
    plot_frontiere(trainx, lambda x: perceptrong.predict(projection_gauss(x, grid_size=grid_size)), 200)
    plot_data(trainx, trainy)

    plt.show()
