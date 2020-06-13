import numpy as np
import pickle
import matplotlib.pyplot as plt
from decisiontree import DecisionTree, entropy, entropy_cond
from tqdm import tqdm


# Une fonction qui décompte un nombre d'éléments dans une liste de np.ndarray donnée
def len_list_nparr(list_nparr):
    return np.sum([np.array(ai).size for ai in list_nparr])


# une méthode qui éstime les probas par comptage des fréquences
def prob_freq(vect):
    vect = np.array(vect)
    return np.unique(vect, return_counts=True)[1] / vect.size


# Q 1.1
# calcul de l'entropie pour les étiquettes données
# si aucunes probas fournies, on utilise celles par défaut
def entropie(vect, prob=prob_freq):
    vect = np.array(vect)
    return -np.sum(prob(vect) * np.log(prob(vect)), axis=0)


# Q 1.2
# calcul de l'entropie conditionnelle à partir de liste des listes et de loi fournies
def entropie_cond(vect_of_vect, prob=prob_freq):
    ec = 0.0
    total_size = len_list_nparr(vect_of_vect)

    # parcours par toutes les listes dans vect_of_vect
    for vect in vect_of_vect:
        pPi = vect.size / total_size
        e = entropie(vect, prob)
        ec += pPi * e
    return ec


def create_partitions(l, pnum, prob=None):
    if prob is None:
        prob = np.ones(pnum) / pnum
    ll = [[] for _ in range(pnum)]
    for elem in l:
        r = np.random.rand()
        ind = np.nonzero(r < np.cumsum(prob))[0][0]
        ll[ind].append(elem)
    return ll


if __name__ == '__main__':

    # Q 1.3
    # print(entropie([-1, -1, -1, -1, 1, 1], prob_freq))
    # data : tableau ( films , features ) , id2titles : dictionnaire id -> titre ,
    # fields : id feature -> nom
    [data, id2titles, fields] = pickle.load(open("imdb_extrait.pkl", "rb"))

    # la derniere colonne est le vote
    datax = data[:, :33]
    datay = np.array([1 if x[33] > 6.5 else -1 for x in data])

    # data_to_p = np.hstack((datax, datay.reshape(-1, 1)))
    # np.random.shuffle(data_to_p)
    # dp = create_partitions(data_to_p, 10)
    # for i in range(len(dp)):
    #     tmp = np.vstack(tuple(dp[:i] + dp[i+1:]))
    # print(sum([len(l) for l in dp]))
    # print(data_to_p.shape[0])

    max_diff, max_entr_attr = -np.inf, -1

    for j in range(datax.shape[1]):
        if np.unique(datax[:, j]).size == 2:
            ypartition = [datay[datax[:, j] == 1], datay[datax[:, j] != 1]]
            e_attr = entropie(datax[:, j]) / np.log(2)
            ec_attr = entropie_cond(ypartition) / np.log(2)

            if max_diff < e_attr - ec_attr:
                max_diff = e_attr - ec_attr
                max_entr_attr = j
            print('Attribut %d, l\'entropie : %f -- %f' %
                  (j, e_attr, entropy(datax[:, j])))
            print('Entropie conditionelle liée à cet attribut : %f -- %f' %
                  (ec_attr, entropy_cond(ypartition)))
            print('Différence entre les entropie au-dessus : %f' % (e_attr - ec_attr))
    print('\nLe meilleur attribut pour commencer c\'est " %s " avec un gain d\'information : %f' %
          (fields[max_entr_attr], max_diff))
    # On constate alors que pour les données fournies " Drama " est un attribut à partir duquel on doit commencer
    # car cela nous donne un gain d'information le plus haut égal à 0.073671
    # On remarque qu'une valeur 0 de l'entropie correspond au cas où il n'y a pas des aspects aléatoires tandis qu'une
    # valeur 1 correspond bien au désordre le plus grand (lorsqu'on a une variable binaire et lorsqu'on prend le log
    # de base 2)
    # Alors, pour le gain d'information des valeurs plus hauts correspondent au cas où le désordre pour des valeurs
    # de l'attribut est plus grand alors que le désordre pour y est minimal. Et cela c'est ce qu'on cherche : un
    # attribut dont les valeurs sont les plus " équilibrées " mais qui produisent les étiquettes les plus " claires "

    sim_num = 5
    depths = [5, 10, 15, 20, 25]
    scores = np.zeros(sim_num)
    split = 2
    for sim in range(sim_num):
        dt = DecisionTree()
        dt.max_depth = depths[sim]  #
        dt.min_samples_split = split  # nombre minimum d ’ exemples pour spliter un noeud
        dt.fit(datax, datay)
        print(dt.score(datax, datay))
        # dessine l ’ arbre dans un fichier pdf si pydot est installe .
        dt.to_pdf("test_tree_%d.pdf" % sim, fields)
        # sinon utiliser http :// www . webgraphviz . com /
        # dt.to_dot(fields)
        # ou dans la console
        # print(dt.print_tree(fields))

        print('Score de classification : %f' % dt.score(datax, datay))
        scores[sim] = dt.score(datax, datay)

    # Q 1.4 des arbres construits sont donnés dans les fichiers " test_tree_%d " où d se varie dans {0, ..., 4} ;
    # ces arbres correspondent aux profondeurs différentes indiquées au-dessus
    # On remarque qu'en fonction de profondeur on sépare de plus en plus exemples lorsque on l'augmente ;
    # Généralement, ce comportement semble normal car à chaque fois qu'on va plus loin dans un arbre, on sépare certains
    # des exemplaires

    figsize = [6.4, 4.8]
    plt.figure(0, figsize=figsize)
    plt.grid(True)
    # plt.title('Évolution de score en fonction de la profondeur')
    plt.xlabel('Profondeur maximale de l\'arbre')
    plt.ylabel('Score de bonne classification')
    plt.plot(depths[:sim_num], scores, 'b-')

    # Q 1.5 Les scores sont fournits dans un graphe créé par matplotlib.pyplot
    # On observe qu'un score de bonne classification augmente si on permet une profondeur de l'arbre plus haut
    # Ce résultat est plutôt attendu donc on peut dire que c'est plutôt normal
    # En revanche, on remarque qu'une croissance de score ralentit dès qu'une profondeur augmente donc l'évolution de
    # score est une fonction concave

    # Q 1.6 En fait, le score ne peut pas être un indicateur fiable du comportement parce que cela nous dit tout
    # simplement que l'algorithme marche bien pour les données fournies en entrée, par contre, il n'y a aucune garantie
    # que cela marcherait pour les autres données pas encore observées. Une approche assez classique pour résoudre ce
    # problème-là c'est de séparer une base de données initiale en deux : celle qu'on utilise pour apprendre le modèle
    # et celle d'autre qu'on n'utilise pas pendant l'apprentissage mais laquelle utilise-t-on pour tester ce modèle
    # obtenu. Notre but est donc de minimiser l'erreur d'apprentissage et celle liée au dataset de test

    # Q 1.7 (le code ci-dessous fournit une réponse demandée)
    #
    # errs = np.zeros((2, sim_num))
    # figsize = [6.4, 3.5]
    # partitions_of_datasets = [(0.8, 0.2), (0.5, 0.5), (0.2, 0.8)]
    # for p in range(len(partitions_of_datasets)):
    #     data_to_p = np.hstack((datax, datay.reshape(-1, 1)))
    #     np.random.shuffle(data_to_p)
    #     train_last_index = int(data_to_p.shape[0] * partitions_of_datasets[p][0]) + 1
    #     datax_train, datax_test = data_to_p[:train_last_index, :-1], data_to_p[train_last_index:, :-1]
    #     datay_train, datay_test = data_to_p[:train_last_index, -1], data_to_p[train_last_index:, -1]
    #     for sim in range(sim_num):
    #         dt = DecisionTree()
    #         dt.max_depth = depths[sim]
    #         dt.min_samples_split = split
    #         dt.fit(datax_train, datay_train)
    #         errs[0, sim] = np.linalg.norm(dt.predict(datax_train) - datay_train)
    #         errs[1, sim] = np.linalg.norm(dt.predict(datax_test) - datay_test)
    #
    #     if p == 0:
    #         plt.figure(1, figsize=figsize)
    #         plt.grid(True)
    #         # plt.title('Évolution de l\'erreur en fonction de la profondeur\n' +
    #         #           'Partition : {0}'.format(partitions_of_datasets[p]))
    #         plt.title('Partition : {0}'.format(partitions_of_datasets[p]))
    #         plt.xlabel('Profondeur maximale de l\'arbre')
    #         plt.ylabel('Erreur')
    #         plt.plot(depths[:sim_num], errs[0], 'b-', label='Erreur d\'apprentissage')
    #         plt.plot(depths[:sim_num], errs[1], 'r-', label='Erreur de test')
    #         plt.legend()
    #     else:
    #         plt.figure(2, figsize=figsize)
    #         plt.subplot(120 + p)
    #         plt.grid(True)
    #         # plt.title('Évolution de l\'erreur en fonction de la profondeur\n' +
    #         #           'Partition : {0}'.format(partitions_of_datasets[p]))
    #         plt.title('Partition : {0}'.format(partitions_of_datasets[p]))
    #         plt.xlabel('Profondeur maximale de l\'arbre')
    #         plt.ylabel('Erreur')
    #         plt.plot(depths[:sim_num], errs[0], 'b-', label='Erreur d\'apprentissage')
    #         plt.plot(depths[:sim_num], errs[1], 'r-', label='Erreur de test')
    #         plt.legend()
    # plt.show()

    # Q 1.8 On constate que quand il y a peu d'exemples d'apprentissage l'erreur d'apprentissage devient la plus petite,
    # en revanche, cela produit l'erreur de test la plus élevée. Pour ce cas-là, tel comportement n'est pas trop
    # inattendue parce que, intuitivement, moins est une taille de données en entrée, plus facile est-il pour le modèle
    # de s'adapter pour ces données. D'autre part, tel modèle n'ajusté que pour le nombre d'exemples assez petit ne peut
    # pas être capable à marcher proprement pour les données pas utilisées pendant l'apprentissage.
    # Pour le cas inverse où la majorité d'exemples sont ceux d'apprentissage, on observe des résultats bien différents.
    # En fait, l'erreur d'apprentissage est la plus élevée par rapport à ces trois partitionnements, par contre, cela
    # nous donne la meilleure l'erreur de test.

    # Q 1.9 Une stabilité des résultats obtenus est une question assez discutable. D'un côté, on constate que le modèle
    # concerné fait assez beaucoup de classifications fautes pour tous les partitionnement, de l'autre côté , cela peut
    # suffire pour les cas particulières où l'exactitude complète n'est pas nécessaire.
    # Par rapport aux améliorations possibles, l'idée de base est de modifier une façon selon laquelle on représente
    # les données. Tout d'abord, cela peut être utile de les normaliser. Ensuite, on pourrait utiliser telle ou telle
    # approche pour extraire des caractéristiques relativement pas principales pour ne pas les considérer pendant une
    # classification. Finalement, on peut utiliser les autres algorithmes de classification pour voir comment cette
    # approche marche par rapport à d'autres.
    print(np.mean(datax, axis=0))

    # Questions de bonus
    pnums = (5, 10, 15, 20, 25)
    data_to_p = np.hstack((datax, datay.reshape(-1, 1)))
    best_err, best_tree, best_params = np.inf, None, None
    for pnum in pnums:
        for sim in range(sim_num):
            print((pnum, depths[sim]))
            np.random.shuffle(data_to_p)
            dp = create_partitions(data_to_p, pnum)
            err = 0.0
            dt = None

            for i in range(len(dp)):
                data_train = np.vstack(tuple(dp[:i] + dp[i+1:]))
                data_test = np.array(dp[i])

                dt = DecisionTree()
                dt.max_depth = depths[sim]
                dt.min_samples_split = split
                dt.fit(data_train[:, :-1], data_train[:, -1])
                err += np.linalg.norm(dt.predict(data_test[:, :-1]) - data_test[:, -1])
            err /= pnum

            if err < best_err:
                best_err = err
                best_tree = dt
                best_params = (pnum, depths[sim])

    print('La meilleure erreur obtenue était : %f,\nproduit par les paramètres pnum = %d, max_depth = %d' %
          (best_err, best_params[0], best_params[1]))
    print('Le meilleur arbre trouvé est fourni dans un fichier best_tree_cross_validation.pdf')
    best_tree.to_pdf('best_tree_cross_validation.pdf', fields)

    # Après avoir relancé des simulations utilisant une validation croisée on constate que l'erreur la plus petite
    # correspond au nombre de partitions le plus grand qui était 25 (parmi {5, 10, 15, 20, 25}). D'autre côté,
    # la profondeur n'était pas maximale et a eu une valeur de 10 (parmi {5, 10, 15, 20, 25}) ce qui est assez
    # intéressant car on attendait plutôt que plus est la profondeur, moins est l'erreur
