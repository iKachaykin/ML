import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from DensityEstimation import histogram, kernel


# plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

# coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48      # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max


def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    # extent pour controler l'echelle du plan


if __name__ == '__main__':
    print(matplotlib.get_backend())

    poidata = pickle.load(open("data/poi-paris.pkl","rb"))
    # liste des types de point of interest (poi)
    print("Liste des types de POI" , ", ".join(poidata.keys()))

    # Choix d'un poi
    typepoi = "night_club"

    # Creation de la matrice des coordonnees des POI
    geo_mat = np.zeros((len(poidata[typepoi]),2))
    rate = np.zeros(len(poidata[typepoi]), dtype=int)
    for i,(k,v) in enumerate(poidata[typepoi].items()):
        geo_mat[i,:]=v[0]
        rate[i] = v[4]

    # Affichage brut des poi
    plt.figure(0, figsize=(15, 7.5))
    plt.subplot(122)
    show_map()
    # alpha permet de regler la transparence, s la taille
    plt.scatter(geo_mat[:, 1], geo_mat[:, 0], alpha=0.8, s=3)

    # On copie geo_mat de manière que dans la copie une première colonne corresponde à x tandis que la deuxième -- à y
    geo_mat_inv = np.zeros_like(geo_mat)
    geo_mat_inv[:, 0] = geo_mat[:, 1]
    geo_mat_inv[:, 1] = geo_mat[:, 0]


    ###################################################

    # discretisation pour l'affichage des modeles d'estimation de densite
    # Nombre de points dans la grille utilisée pour l'affichage de la densité
    steps = 200
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, steps), np.linspace(ymin, ymax, steps))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # print(grid)

    # Nombre de points dans la grille utilisée pour l'apprentissage de la densité
    # On vous recommande d'utiliser des grilles différents pour l'apprentissage et pour l'affichage afin d'assurer une
    # image plus précise de la densité obtenue, de plus, on vous recommande de vérifier une condition suivante :
    # steps > steps_appr
    steps_appr = 50
    monModele = histogram(geo_mat_inv, (xmin, xmax, ymin, ymax), (steps_appr, steps_appr))
    # monModele = kernel(geo_mat_inv, (xmin, xmax, ymin, ymax), (steps_appr, steps_appr), mode='gauss')
    res = monModele.predict(grid).reshape(steps, steps)
    # A remplacer par res = monModele.predict(grid).reshape(_steps,_steps)
    # res = np.random.random((steps, steps))
    plt.subplot(121)
    # plt.figure()
    # show_map()
    plt.imshow(res, extent=[xmin, xmax, ymin, ymax], interpolation='none', alpha=0.3, origin="lower", aspect=1.5)
    plt.colorbar()
    plt.scatter(geo_mat[:, 1], geo_mat[:, 0], alpha=0.3, s=3)
    plt.show()
