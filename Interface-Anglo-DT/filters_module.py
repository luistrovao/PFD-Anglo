import pandas as pd
import numpy as np

class filtros():
    def __init__(self):
        self.funcoes = {
            "nao_numerico": self.nao_num,
            "quartiles": self.quartiles,
            "clusterizacao": self.clusters,
            "bases_c":self.base_consumo,
            "bases_temp": self.base_temps,
            "bases_esc": self.base_esc,
            "nao-negativo": self.nao_negativo,
            "retira_virgula": self.virgula,
        }

    def nao_negativo(self,base):
        base = base[(base >= 0).all(1)]
        return base

    def base_consumo(self, base):

        base_consumo = base.drop(columns = ["Temp. Blocos Vazamento Metal ( bica 04)",
                                         "Temp. Blocos Vazamento Metal ( bica 03)",
                                         "Temp. Free Board",
                                         "Temp. Blocos Vazamento Escória ( bica 01)",
                                         "Temp. Blocos Vazamento Escória ( bica 02)",
                                         "Temp. Blocos Vazamento Escória ( bica 03)",
                                         "Temp. Blocos Vazamento Escória ( bica 04)",
                                         "Temp. Blocos Vazamento Escória ( bica 05)",
                                         "Temp. Blocos Vazamento Escória ( bica 06)",
                                         "Temp. da Escória",
                                         "Potencia_Ativa_Total"], axis = 1)

        a = pd.DataFrame(np.zeros(shape=(1,0))) # Linha com zeros adicionada ao fim

        # Seleciona Coluna da Potência
        Potencia = pd.DataFrame(base.loc[:, 'Potencia_Ativa_Total'])
        Potencia.reset_index(drop=True, inplace=True)

        #Ajuste para utilizar valores médios
        base_aux = base_consumo.drop(base.index[0])
        base_aux = base_aux.append(a)
        base_aux.fillna(0)
        base_aux.reset_index(drop=True, inplace=True)

        # Cria dataframe auxiliar para subtração da linha com a posterior
        P_aux = Potencia.drop(Potencia.index[0])
        P_aux = P_aux.append(a)
        P_aux.fillna(0)
        P_aux.reset_index(drop=True, inplace=True)

        Soma = base_consumo.add(base_aux, fill_value=0).div(2)
        Consumo = P_aux.subtract(Potencia, fill_value=0)
        Soma['Potencia_Ativa_Total'] = Consumo
        base['Potencia_Ativa_Total'] = Consumo

        base_consumo = Soma
        base_consumo.drop(base_consumo.tail(1).index, inplace=True)

        return base_consumo

    def base_temps(self, base):
        pass

    def base_esc(self,base):
        pass

    def nao_num(self, base):
        # Converte para Dataframe
        DF = pd.DataFrame(base)
        # Converte String para NaN
        DF[DF.columns[1:len(DF.columns)]] = DF[DF.columns[1:len(DF.columns)]].apply(pd.to_numeric, errors='coerce')
        # Apaga as linhas com valores NaN
        DF = DF.dropna()
        # Reseta os indices do novo dataframe
        DF.reset_index(drop=True, inplace=True)

        return DF

    def quartiles(self, base):
        DF = pd.DataFrame(base)
        columns = list(DF)

        Q1 = DF.quantile(0.25, axis=0, numeric_only=True, interpolation='linear')
        Q3 = DF.quantile(0.75, axis=0, numeric_only=True, interpolation='linear')
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR

        return lim_sup, lim_inf

    def clusters(self, n_clusters, base):
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)

        kmeans.fit(base)

        return kmeans.labels_, kmeans

    def virgula(self,texto):
        valor = texto
        if texto.find(',') != -1:
            valor = texto.replace(",", ".")

        return valor
