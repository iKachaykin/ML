import warnings
import numpy as np
from numpy import ndarray
from tqdm import tqdm


def mse(X, y, w):
    """
    Mean squared error.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples of objects' features where *n_samples* is a number of samples and *n_features* is a number of features.
    y : array-like of shape(n_samples,)
        Vector of labels corresponding to *X*.
    w : array-like of shape(n_features,)
        Vector of weights used to make a prediction.

    Returns
    -------
    mse : float
        Value of the mean squared error computed for the given parameters.

    """
    X = np.array(X)
    if len(X.shape) == 1:
        X = np.array([X])
    y = np.array(y)
    if len(y.shape) == 0:
        y = np.array([y])
    return np.mean((y - np.dot(X, w)) ** 2)


def mse_g(X, y, w):
    """
    Gradient of the mean squared error.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples of objects' features where *n_samples* is a number of samples and *n_features* is a number of features.
    y : array-like of shape(n_samples,)
        Vector of labels corresponding to *X*.
    w : array-like of shape(n_features,)
        Vector of weights used to make a prediction.

    Returns
    -------
    mse_g : ndarray of shape (n_features,)
        Gradient of the mean squared error by the variable *w* computed for the given parameters.
    """
    X = np.array(X)
    if len(X.shape) == 1:
        X = np.array([X])
    y = np.array(y)
    if len(y.shape) == 0:
        y = np.array([y])
    return -2 * np.mean((y - np.dot(X, w)).reshape(-1, 1) * X, axis=0)


def mse_regL1(X, y, w, alpha=1.0):
    """
    Mean squared error regularized with the penalty L1.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples of objects' features where *n_samples* is a number of samples and *n_features* is a number of features.
    y : array-like of shape(n_samples,)
        Vector of labels corresponding to *X*.
    w : array-like of shape(n_features,)
        Vector of weights used to make a prediction.
    alpha : float, default=1.0
        Parameter of regularization.

    Returns
    -------
    mse_reg : float
        Regularized value of the mean squared error computed for the given parameters with the penalty L1.

    """
    return mse(X, y, w) + alpha * np.sum(np.abs(w))


def mse_regL1_g(X, y, w, alpha=1.0):
    """
    Gradient of the mean squared error regularized with the penalty L1.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples of objects' features where *n_samples* is a number of samples and *n_features* is a number of features.
    y : array-like of shape(n_samples,)
        Vector of labels corresponding to *X*.
    w : array-like of shape(n_features,)
        Vector of weights used to make a prediction.
    alpha : float, default=1.0
        Parameter of regularization.

    Returns
    -------
    mse_reg_g : ndarray of shape (n_features,)
        Gradient by the variable *w* of the mean squared error regularized with the penalty L1 computed for the given
        parameters.
    """
    return mse_g(X, y, w) + alpha * np.sign(w)


def mse_regL2(X, y, w, alpha=1.0):
    """
    Mean squared error regularized with the penalty L2.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples of objects' features where *n_samples* is a number of samples and *n_features* is a number of features.
    y : array-like of shape(n_samples,)
        Vector of labels corresponding to *X*.
    w : array-like of shape(n_features,)
        Vector of weights used to make a prediction.
    alpha : float, default=1.0
        Parameter of regularization.

    Returns
    -------
    mse_reg : float
        Regularized value of the mean squared error computed for the given parameters with the penalty L2.

    """
    return mse(X, y, w) + alpha * np.linalg.norm(w) ** 2


def mse_regL2_g(X, y, w, alpha=1.0):
    """
    Gradient of the mean squared error regularized with the penalty L2.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples of objects' features where *n_samples* is a number of samples and *n_features* is a number of features.
    y : array-like of shape(n_samples,)
        Vector of labels corresponding to *X*.
    w : array-like of shape(n_features,)
        Vector of weights used to make a prediction.
    alpha : float, default=1.0
        Parameter of regularization.

    Returns
    -------
    mse_reg_g : ndarray of shape (n_features,)
        Gradient by the variable *w* of the mean squared error regularized with the penalty L2 computed for the given
        parameters.
    """
    return mse_g(X, y, w) + 2 * alpha * w


class RegressionClassifier:

    """
    Class representing regression classifier
    """

    def __init__(
            self, alpha=1.0, loss='linear', loss_g=None, max_iter=1000, eps=0.01, use_tqdm=False, wtol=1e-4, ltol=1e-4,
            gtol=1e-4, min_iter=0, modefit='batch', modstep=False, mbnum=None, ignore_warnings=False
    ):
        self.alpha = alpha
        if loss == 'linear':
            self.loss, self.loss_g = mse, mse_g
        elif loss == 'ridge':
            self.loss, self.loss_g = \
                lambda X, y, w: mse_regL2(X, y, w, alpha), lambda X, y, w: mse_regL2_g(X, y, w, alpha)
        elif loss == 'lasso':
            self.loss, self.loss_g = \
                lambda X, y, w: mse_regL1(X, y, w, alpha), lambda X, y, w: mse_regL1_g(X, y, w, alpha)
        elif callable(loss) and callable(loss_g):
            self.loss, self.loss_g = loss, loss_g
        else:
            raise ValueError('"loss" type not understood!')
        self.max_iter = max_iter
        self.eps = eps
        self.use_tqdm = use_tqdm
        self.wtol = wtol
        self.ltol = ltol
        self.gtol = gtol
        self.min_iter = min_iter
        if modefit == 'batch':
            self._epoch_step = self._epoch_step_batch
        elif modefit == 'sgd':
            self._epoch_step = self._epoch_step_sgd
        elif modefit == 'mini-batch':
            self._epoch_step = self._epoch_step_mini_batch
        elif callable(modefit):
            self._epoch_step = modefit
        else:
            raise ValueError('"modefit" type not understood!')
        self.modefit = modefit if isinstance(modefit, str) else 'external'
        self.modstep = modstep
        if modstep and self.modefit != 'batch' and not ignore_warnings:
            warnings.warn('"modstep" was True but would be never used because "modefit" != "batch"')
        self.mbnum = mbnum
        if mbnum is not None and self.modefit != 'mini-batch' and not ignore_warnings:
            warnings.warn('"mbnum" was passed but would be never used because "modefit" != "mini-batch"')

        self.w = None
        self.wlist = []
        self.losses = {'train': [], 'test': []}

    def fit(self, X, y, testX=None, testy=None):
        self.wlist = []
        if testX is not None and testy is not None:
            self.losses['test'] = []
            self.losses['train'] = []
            testX = np.hstack((testX, np.ones((len(testy), 1))))
            lstats = True
        else:
            lstats = False
        N = len(y)
        X = np.array(X).reshape(N, -1)
        D = X.shape[1]
        # On ajoute une colonne de 1 pour considérer un bias
        X = np.hstack((X, np.ones((N, 1))))
        self.w = np.random.rand(D+1)
        g = self.loss_g(X, y, self.w)
        step = self.eps
        self.wlist = [self.w.copy()]
        if self.use_tqdm:
            iters = tqdm(range(self.max_iter))
        else:
            iters = range(self.max_iter)
        for i in iters:
            wpred, lpred, gpred = self.w.copy(), self.loss(X, y, self.w), g.copy()
            if lstats:
                self.losses['train'].append(lpred)
                self.losses['test'].append(self.loss(testX, testy, self.w))

            self.w, g, step = self._epoch_step(X, y, self.w, g, step, self.loss_g, wpred, gpred)
            self.wlist.append(self.w.copy())

            if (i >= self.min_iter and np.linalg.norm(self.w - wpred) < self.wtol and
                    np.abs(self.loss(X, y, self.w) - lpred) < self.ltol and (
                            np.linalg.norm(g) < self.gtol or
                            (np.linalg.norm(g - gpred) < self.gtol if self.modefit == 'batch' else False)
                    )
            ):
                print(i)
                break
        if lstats:
            self.losses['train'].append(self.loss(X, y, self.w))
            self.losses['test'].append(self.loss(testX, testy, self.w))

    def _epoch_step_batch(self, X, y, w, g, step, loss_g, wpred, gpred):
        w = w - step * g
        g = loss_g(X, y, w)
        if self.modstep:
            step = (
                np.abs(np.dot(w - wpred, g - gpred)) / np.linalg.norm(g - gpred) ** 2
                if np.all(g != gpred) else 0.0)
        return w, g, step

    def _epoch_step_sgd(self, X, y, w, g, step, loss_g, wpred=None, gpred=None):
        data_shuffled = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(data_shuffled)
        X_shuffled, y_shuffled = data_shuffled[:, :-1], data_shuffled[:, -1]
        for xline, yline in zip(X_shuffled, y_shuffled):
            w = w - step * loss_g(xline, yline, w)
        return w, loss_g(X, y, w), step

    def _epoch_step_mini_batch(self, X, y, w, g, step, loss_g, wpred=None, gpred=None):
        data_stacked = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(data_stacked)
        batches = [[] for _ in range(self.mbnum)]
        for dline in data_stacked:
            bind = np.random.randint(0, self.mbnum, 1)[0]
            batches[bind].append(dline)
        for bind in range(self.mbnum):
            batches[bind] = np.array(batches[bind])

        for batch in batches:
            w = w - step * loss_g(batch[:, :-1], batch[:, -1], w)

        return w, loss_g(X, y, w), step

    def predict(self, X):
        """
        On prédit des labels pour les données passées

        :param X: données pour lesquelles il faut prédire des labels
        :return yprn: prédiction des labels
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        N = X.shape[0]
        X = np.hstack((X, np.ones((N, 1))))
        return np.sign(np.dot(X, self.w))

    def score(self, X, y):
        """
        Calcul du pourcentage de bonne classification

        :param X: données pour lesquelles on prédit des labels
        :param y: vrais labels de données passées
        :return score: un nombre normalisée sur [0, 1] de bonnes classifications
        """
        return np.mean(np.where(y - self.predict(X) == 0, 1, 0))

    def evaluate(self, trainX, trainy, testX, testy, weights=None):
        score_train, score_test = self.score(trainX, trainy), self.score(testX, testy)
        if weights is None:
            weights = np.ones(3)
        return 0.5 * np.dot(weights, [score_train, score_test, -np.abs(score_train - score_test)])
