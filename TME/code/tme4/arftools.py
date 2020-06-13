import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
from tqdm import tqdm


def is_sorted(l):
    """ Vérification si une liste passée est triée ou pas """
    return all(l[i] <= l[i + 1] for i in range(len(l) - 1))


def plot_data(data,labels=None):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    cols,marks = ["red", "green", "blue", "orange", "black", "cyan"],[".","+","*","o","x","^"]
    if labels is None:
        plt.scatter(data[:,0],data[:,1],marker="x")
        return
    for i,l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels==l,0],data[labels==l,1],c=cols[i],marker=marks[i])

def plot_frontiere(data,f,step=20):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),colors=('gray','blue'),levels=[-1,0,1])

def plot_frontiere_proba(data,f,step=20):
    """ Trace un graphe de la frontiere de decision de f pour le cas de proba
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape), 255)
    plt.colorbar()

def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type==0:
         #melange de 2 gaussiennes
         xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//2)
         xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//2)
         data=np.vstack((xpos,xneg))
         y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))

    if data_type==2:
        #echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1
    # un peu de bruit
    data[:,0]+=np.random.normal(0,epsilon,nbex)
    data[:,1]+=np.random.normal(0,epsilon,nbex)
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data=data[idx,:]
    y=y[idx]
    return data,y

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
    """ retourne la moyenne de l'erreur hinge """
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


def reg_hinge(datax, datay, w, bias=0.0, alpha=0.0001):
    """ retourne la moyenne de l'erreur hinge regularisée """
    datax = np.array(datax)
    if len(datax.shape) == 1:
        datax = np.array([datax])
    datay = np.array(datay)
    if len(datay.shape) == 0:
        datay = np.array([datay])
    return np.mean(np.maximum(0.0, bias - datay * np.dot(datax, w))) + alpha * np.linalg.norm(w) ** 2


def reg_hinge_g(datax, datay, w, bias=0.0, alpha=0.0001):
    """ retourne le gradient moyen de l'erreur hinge regularisée """
    """ retourne le gradient moyen de l'erreur hinge """
    datax = np.array(datax)
    if len(datax.shape) == 1:
        datax = np.array([datax])
    datay = np.array(datay)
    if len(datay.shape) == 0:
        datay = np.array([datay])
    return np.mean(np.where(
        (bias - datay * np.dot(datax, w)).reshape(-1, 1) * np.ones_like(datax) > 0,
        -datay.reshape(-1, 1) * datax, 0.0),
        axis=0
    ) + 2 * alpha * w


def evaluate_scores(train_score, test_score):
    return np.sum([train_score, test_score, -np.abs(train_score - test_score)])


class Lineaire(object):
    def __init__(self, loss=hinge, loss_g=hinge_g, max_iter=1000, eps=0.01, use_tqdm=False, bias=None, alpha=None):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter, eps
        if bias is None or alpha is None:
            self.loss, self.loss_g = loss, loss_g
        else:
            self.loss, self.loss_g = lambda datax, datay, w: loss(datax, datay, w, bias, alpha), \
                                     lambda datax, datay, w: loss_g(datax, datay, w, bias, alpha)
        self.use_tqdm = use_tqdm
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
        if self.use_tqdm:
            iters = tqdm(range(self.max_iter))
        else:
            iters = range(self.max_iter)
        for i in iters:
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
        if self.use_tqdm:
            iters = tqdm(range(self.max_iter))
        else:
            iters = range(self.max_iter)
        for i in iters:
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
        if self.use_tqdm:
            iters = tqdm(range(self.max_iter))
        else:
            iters = range(self.max_iter)
        for i in iters:
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
