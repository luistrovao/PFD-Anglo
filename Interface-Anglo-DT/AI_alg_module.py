from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

import numpy as np
import pandas as pd

class algoritmos_AI():
    def __init__(self):
        self.tipos = {
            "Dec_Tree": self.decision_tree,
        }

    def decision_tree(self, base, max_dep, n_estim, n_clusters, entradas, saidas):

        self.classes = []
        self.df = base

        i = 0
        while i < n_clusters:
            self.classes.append(self.df.loc[self.df['K classes'] == i])
            i += 1

        self.df = self.df.drop('K classes', axis=1)


        dados = []
        i = 0
        previsoes = []
        resultados = []
        modelo = []
        k = 0
        N_inputs = len(entradas)
        N_outputs = len(saidas)


        while i < n_clusters:
            dados.append(train_test_split(self.classes[i], test_size=0.2))
            modelo.append([0] * N_outputs)
            previsoes.append([0] * N_outputs)
            resultados.append([0] * N_outputs)

            while k < N_outputs:
                regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_dep),
                                         n_estimators=n_estim, random_state=None)
                modelo[i][k] = regr.fit(dados[i][0].loc[:, entradas], dados[i][0].loc[:, saidas[k]])
                previsoes[i][k] = regr.predict(dados[i][1].loc[:, entradas])
                resultados[i][k] = np.mean(100 * abs(np.asarray(dados[i][1].loc[:, saidas[k]]) -
                                                     previsoes[i][k]) / np.asarray(dados[i][1].loc[:, saidas[k]]))
                k += 1

            k = 0
            i += 1

        return dados, previsoes, resultados, modelo
