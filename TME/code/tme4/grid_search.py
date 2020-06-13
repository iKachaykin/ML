from arftools import *
import numpy as np
from sklearn import svm
import warnings
from tqdm import tqdm

if __name__ == '__main__':
    # Grid search
    data_type = 0
    noise = 1.0
    max_iter = 1000
    warnings.filterwarnings("ignore")

    kernels = ('linear', 'poly', 'rbf', 'sigmoid')
    nbexs = (1000, 2000, 3000, 4000, 5000)
    all_params = ('C', 'degree', 'gamma', 'coef0', 'shrinking', 'probability')
    # all_params_adm = {
    #     'C': list(np.linspace(0, 2, 11)[1:]),
    #     'degree': list(np.arange(2, 6)),
    #     'gamma': list(np.linspace(0, 2, 11)[1:]) + ['auto', 'scale'],
    #     'coef0': list(np.linspace(-2, 2, 21)),
    #     'shrinking': [True, False],
    #     'probability': [True, False]
    # }
    all_params_adm = {
        'C': [0.5, 1.0, 2.0],
        'degree': [3, 4, 5],
        'gamma': ['auto', 'scale', 1.0],
        'coef0': [-1.0, 0.0, 1.0],
        'shrinking': [True, False],
        'probability': [True, False]
    }

    best_found_params = {}
    for kernel in kernels:
        for nbex in tqdm(nbexs):
            trainx, trainy = gen_arti(nbex=nbex, data_type=data_type, epsilon=noise)
            testx, testy = gen_arti(nbex=nbex, data_type=data_type, epsilon=noise)
            best_found_params[(kernel, nbex)] = {'target': -np.inf}
            for probability in all_params_adm['probability']:
                for shrinking in all_params_adm['shrinking']:
                    for C in all_params_adm['C']:
                        if kernel in ['rbf', 'poly', 'sigmoid']:
                            for gamma in all_params_adm['gamma']:
                                if kernel in ['poly', 'sigmoid']:
                                    for coef0 in all_params_adm['coef0']:
                                        if kernel == 'poly':
                                            for degree in all_params_adm['degree']:
                                                svc = svm.SVC(
                                                    kernel=kernel, max_iter=max_iter, probability=probability,
                                                    shrinking=shrinking, C=C, gamma=gamma, coef0=coef0, degree=degree)
                                                svc.fit(trainx, trainy)
                                                train_score, test_score = \
                                                    svc.score(trainx, trainy), svc.score(testx, testy)
                                                target = evaluate_scores(train_score, test_score)
                                                if target > best_found_params[(kernel, nbex)]['target']:
                                                    best_found_params[(kernel, nbex)]['probability'] = probability
                                                    best_found_params[(kernel, nbex)]['shrinking'] = shrinking
                                                    best_found_params[(kernel, nbex)]['C'] = C
                                                    best_found_params[(kernel, nbex)]['gamma'] = gamma
                                                    best_found_params[(kernel, nbex)]['coef0'] = coef0
                                                    best_found_params[(kernel, nbex)]['degree'] = degree
                                                    best_found_params[(kernel, nbex)]['train_score'] = train_score
                                                    best_found_params[(kernel, nbex)]['test_score'] = test_score
                                                    best_found_params[(kernel, nbex)]['target'] = target
                                        else:
                                            svc = svm.SVC(
                                                kernel=kernel, max_iter=max_iter, probability=probability,
                                                shrinking=shrinking, C=C, gamma=gamma, coef0=coef0)
                                            svc.fit(trainx, trainy)
                                            train_score, test_score = svc.score(trainx, trainy), svc.score(testx, testy)
                                            target = evaluate_scores(train_score, test_score)
                                            if target > best_found_params[(kernel, nbex)]['target']:
                                                best_found_params[(kernel, nbex)]['probability'] = probability
                                                best_found_params[(kernel, nbex)]['shrinking'] = shrinking
                                                best_found_params[(kernel, nbex)]['C'] = C
                                                best_found_params[(kernel, nbex)]['gamma'] = gamma
                                                best_found_params[(kernel, nbex)]['coef0'] = coef0
                                                best_found_params[(kernel, nbex)]['train_score'] = train_score
                                                best_found_params[(kernel, nbex)]['test_score'] = test_score
                                                best_found_params[(kernel, nbex)]['target'] = target
                                else:
                                    svc = svm.SVC(
                                        kernel=kernel, max_iter=max_iter, probability=probability,
                                        shrinking=shrinking, C=C, gamma=gamma)
                                    svc.fit(trainx, trainy)
                                    train_score, test_score = svc.score(trainx, trainy), svc.score(testx, testy)
                                    target = evaluate_scores(train_score, test_score)
                                    if target > best_found_params[(kernel, nbex)]['target']:
                                        best_found_params[(kernel, nbex)]['probability'] = probability
                                        best_found_params[(kernel, nbex)]['shrinking'] = shrinking
                                        best_found_params[(kernel, nbex)]['C'] = C
                                        best_found_params[(kernel, nbex)]['gamma'] = gamma
                                        best_found_params[(kernel, nbex)]['train_score'] = train_score
                                        best_found_params[(kernel, nbex)]['test_score'] = test_score
                                        best_found_params[(kernel, nbex)]['target'] = target
                        else:
                            svc = svm.SVC(
                                kernel=kernel, max_iter=max_iter, probability=probability,
                                shrinking=shrinking, C=C)
                            svc.fit(trainx, trainy)
                            train_score, test_score = svc.score(trainx, trainy), svc.score(testx, testy)
                            target = evaluate_scores(train_score, test_score)
                            if target > best_found_params[(kernel, nbex)]['target']:
                                best_found_params[(kernel, nbex)]['probability'] = probability
                                best_found_params[(kernel, nbex)]['shrinking'] = shrinking
                                best_found_params[(kernel, nbex)]['C'] = C
                                best_found_params[(kernel, nbex)]['train_score'] = train_score
                                best_found_params[(kernel, nbex)]['test_score'] = test_score
                                best_found_params[(kernel, nbex)]['target'] = target

    print('Les meilleurs paramètres trouvés\n\n')
    for kernel in kernels:
        for nbex in nbexs:
            print('Kernel : {0}, nbex : {1},\nparams : {2}'.format(kernel, nbex, best_found_params[(kernel, nbex)]))
