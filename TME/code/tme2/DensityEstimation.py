import numpy as np
import warnings
from numpy import ndarray


def kernel_parzen(x):
    """
    Fenêtre de Parzen.

    Parameters
    ----------
    x : ndarray
        Plusieurs points où il faut calculer des valeurs du noyau.

    Returns
    -------
    vals : ndarray
        Tableau avec des valeurs correspondants du noyau.
    """
    return np.array(np.all(np.abs(np.array(x)) <= 0.5, axis=(len(np.array(x).shape)-1)))


def kernel_gauss(x):
    """
    Noyau gaussienne.

    Parameters
    ----------
    x : ndarray
        Plusieurs points où il faut calculer des valeurs du noyau.

    Returns
    -------
    vals : ndarray
        Tableau avec des valeurs correspondants du noyau.
    """
    laxis = len(np.array(x).shape)-1
    dim = np.array(x).shape[-1]
    return np.array(
        (2*np.pi) ** (-dim/2) * np.exp(-0.5*np.linalg.norm(x, axis=laxis)**2)
    )


def kernel_wrapper(x, points, hclength, mode, kernel_custom=None):
    """
    Un wrapper des noyaux différents qui les utilise pour calculer la densité.

    Parameters
    ----------
    x : ndarray
        Point où il faut calculer une valeur de la densité.
    points : ndarray
        Points de l'échantillon.
    hclength : float
        Longeur de l'hypercube où on suppose la valeur constante de la densité. Si rien a été soumis, alors on la
        calcule comme la longeur moyenne de la case dans la grille passée.
    mode : str
        Type de noyau à utiliser ; peut prendre une valeur 'gauss' pour gaussienne, 'parzen' pour la fenêtre de Parzen,
        ou 'custom' pour un noyau défini par l'utilisateur et passé dans un paramètre *kernel_custom*.
    kernel_custom : collections.abc.Callable
        Noyau défini par l'utilisateur dans le cas où on a passé *mode* = 'custom'. Cet objet doit surtout être défini
        de manière que kernel_custom(points : ndarray) -> vals : ndarray, où *points* est un tableau avec des plusieurs
        points de l'espace de dimension quelconque et *vals* est un tableau avec des valeurs correspondantes de noyau.

    Returns
    -------
    dens : float
        Valeur de la densité dans un point x.
    """
    _x_c = np.array(x)
    _points_c = np.array(points)
    if mode == 'gauss':
        fkernel = kernel_gauss
    elif mode == 'parzen':
        fkernel = kernel_parzen
    elif mode == 'custom':
        fkernel = kernel_custom
    else:
        raise ValueError('mode variable was not recognized!')
    return np.sum(fkernel((_x_c-_points_c)/hclength)) / _points_c.shape[0] / hclength ** _points_c.shape[1]


def point_in_rect(point, rect):
    """
    Détermine si le point passé appartient à un rectangle soumis.

    Parameters
    ----------
    point : ndarray / list
        Point pour lequel il faut effectuer une vérification.
    rect : ndarray / list / tuple
        Rectangle pour lequel il faut effectuer une vérification.

    Returns
    -------
    ind : bool
        True si *point* appartient à *rect*, False sinon.
    """
    _rect_reshaped = np.array(rect).reshape(-1, 2)
    _point_c = np.array(point)
    return np.all(np.logical_and(_rect_reshaped[:, 0] <= _point_c, _point_c <= _rect_reshaped[:, 1]))


def get_point_inds(point, ranges):
    """
    Retourne des index dans la grille du points passé.

    Parameters
    ----------
    point : ndarray / list
        Point pour lequel on trouve des index.
    ranges : ndarray
        Grilles indépendantes des points.

    Returns
    -------
    inds : tuple
        Des index qu'il faut retourner.
    """
    _ranges_c = np.array(ranges)
    _point_c = np.array(point)

    inds = np.argmin(
        np.where(_ranges_c - _point_c.reshape(-1, 1) >= 0, _ranges_c - _point_c.reshape(-1, 1), np.inf), axis=1
    ) - 1

    return tuple(np.array(np.maximum(inds, 0), dtype=int))


def histogram(points, rect, steps):
    """
    Éstimateur de densité pas la méthode des histogrammes.

    Parameters
    ----------
    points : ndarray / list
        L'échantillon duquel on doit apprendre la densité.
    rect : tuple(float) = (xmin, xmax, ymin, ymax, ...) / ndarray
        Rectangle où on éstime la densité.
    steps : tuple(int) = (xsteps, ysteps, ...)
        Nombre des pas sur chaque axe.

    Returns
    -------
    model : DensityEstimationModel
        Un modèle appris depuis les données passées.
    """
    _rect_c = np.array(rect).reshape(-1, 2)
    _points_c = np.array(points)
    ranges = np.array([
        np.linspace(varmin, varmax, varstep)
        for varmin, varmax, varstep in zip(_rect_c[:, 0], _rect_c[:, 1], steps)
    ])
    density_table = np.zeros(steps)
    for point in _points_c:
        density_table[get_point_inds(point, ranges)] += 1
    density_table = density_table / (_points_c.shape[0] * np.prod(
        [(varmax-varmin)/(varstep-1) for varmin, varmax, varstep in zip(_rect_c[:, 0], _rect_c[:, 1], steps)])
    )
    return DensityEstimationModel(rect, steps, density_table=density_table)


def kernel(points, rect, steps, hclength=None, mode='gauss', kernel_custom=None):
    """
    Éstimateur de densité par la méthode à noyaux.

    Parameters
    ----------
    points : ndarray
        L'échantillon duquel on doit apprendre la densité.
    rect : tuple(float) = (xmin, xmax, ymin, ymax, ...) / ndarray
        Rectangle où on éstime la densité.
    steps : tuple(int) = (xsteps, ysteps, ...)
        Nombre des pas sur chaque axe.
    hclength : float
        Longeur de l'hypercube où on suppose la valeur constante de la densité. Si rien a été soumis, alors on la
        calcule comme la longeur moyenne de la case dans la grille passée.
    mode : str
        Type de noyau à utiliser ; peut prendre une valeur 'gauss' pour gaussienne, 'parzen' pour la fenêtre de Parzen,
        ou 'custom' pour un noyau défini par l'utilisateur et passé dans un paramètre *kernel_custom*.
    kernel_custom : collections.abc.Callable
        Noyau défini par l'utilisateur dans le cas où on a passé *mode* = 'custom'. Cet objet doit surtout être défini
        de manière que kernel_custom(points : ndarray) -> vals : ndarray, où *points* est un tableau avec des plusieurs
        points de l'espace de dimension quelconque et *vals* est un tableau avec des valeurs correspondantes de noyau.

    Returns
    -------
    model : DensityEstimationModel
        Un modèle appris depuis les données passées.
    """
    _rect_c = np.array(rect).reshape(-1, 2)
    if hclength is None:
        hclength = np.mean((_rect_c[:, 1] - _rect_c[:, 0]) / (np.array(steps) - 1))

    if kernel_custom is not None and mode != 'custom':
        warnings.warn('kernel_custom was passed but would not be used because mode was not equal to "custom"!')

    def density_function(x):
        return kernel_wrapper(x, points, hclength, mode.lower(), kernel_custom=kernel_custom)

    return DensityEstimationModel(rect, steps, density_function=density_function)


class DensityEstimationModel:
    """
    Représente un modèle de l'éstimation de la densité.

    Parameters
    ----------
    rect : tuple(float) = (xmin, xmax, ymin, ymax, ...) / ndarray
        Rectangle où on éstime la densité.
    steps : tuple(int) = (xsteps, ysteps, ...)
        Nombre des pas sur chaque axe.
    density_table : ndarray, facultatif
        Table qui contient des valeurs de la densité dans les points de la grille.
    density_function : collections.abc.Callable, facultatif
        Fonction estimée qui représente la densité du modèle.

    Attributes
    ---------
    _rect : ndarray
        Rectangle où on éstime la densité.
    _steps : tuple(int) = (xsteps, ysteps, ...)
        Nombres des pas sur chaque axe.
    _ranges : ndarray
        Tableau qui stocke des grilles indépendantes de chaque variable.
    _density_table : ndarray
        Table qui contient des valeurs de la densité dans les points de la grille. Attribut qui ne peut pas être utilisé
        en même temps avec un attribut *self._density_function*.
    _density_function : collections.abc.Callable
        Fonction estimée qui représente la densité du modèle. Attribut qui ne peut pas être utilisé en même temps avec
        un attribut *self._density_table*.
    """

    def __init__(self, rect, steps, density_table=None, density_function=None):
        if (
                (density_table is None and density_function is None) or
                (density_table is not None and density_function is not None)
        ):
            raise ValueError('Il faut y mettre exactement un paramètre qui est soit une table de densité, soit une'
                             'fonction sous en forme explicite')
        if len(np.array(rect).shape) == 2 and np.array(rect).shape[1] == 2:
            self._rect = np.copy(rect)
        else:
            self._rect = np.array(rect).reshape(-1, 2)
        self._steps = steps
        self._ranges = np.array([
            np.linspace(varmin, varmax, varstep)
            for varmin, varmax, varstep in zip(self._rect[:, 0], self._rect[:, 1], self._steps)
        ])
        if density_table is not None:
            self._density_table = density_table
            self._density_function = None
        if density_function is not None:
            self._density_function = density_function
            self._density_table = None

    def predict(self, points):
        """
        Prédit la densité des points passés.

        Parameters
        ----------
        points : ndarray / list
            Points dont la densité il faut prédire.

        Returns
        -------
        dens : ndarray
            Densités des points passés.
        """
        _points_c = np.array(points)
        if len(_points_c.shape) == 1:
            return self._predict_in_point(_points_c)
        dens = np.zeros(_points_c.shape[0])
        for i in range(_points_c.shape[0]):
            dens[i] = self._predict_in_point(_points_c[i])
        return dens

    def _predict_in_point(self, point):
        """
        Prédit la densité dans un seul point passé.

        Parameters
        ----------
        point : ndarray / list
            Point dont la densité il faut prédire.

        Returns
        -------
        dens : float
            Densité du point passé.
        """
        _point_c = np.copy(point)
        if not point_in_rect(_point_c, self._rect):
            return 0.0
        if self._density_table is not None:
            inds = get_point_inds(point, self._ranges)
            return self._density_table[inds]
        if self._density_function is not None:
            return self._density_function(point)
