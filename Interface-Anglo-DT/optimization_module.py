# coding: utf-8
from scipy.optimize import minimize


class otimizacao():
    def __init__(self,Inputs_t,Inputs_c,kmeans,Models,x0,lb,ub):
        self.inputs_t = Inputs_t
        self.inputs_c = Inputs_c
        self.kmeans = kmeans
        self.models = Models
        self.x0 = x0
        self.lb = lb
        self.ub = ub
        self.methods = {
            "Consumo": self.calcConsumo,
            "Desarme": self.calcDesarme,
            "Funcao_objetivo": self.objective,
            "minimizar":self.minimizacao(),
        }

    def calcConsumo(self,x):
            
        electric_variables = [x[0],x[1],x[2],x[3],x[4],x[5]]
        #verificar a necessidade de converter eletric para DF
        
        input_consumo =  self.inputs_c + electric_variables
        #input_consumo = pd.concat[Inputs,eletrics_variables,axis = 1]
        cluster = self.kmeans.predict(input_consumo)
       
        consumo = self.models[1][cluster[0]].predict(input_consumo)
     
        return consumo


    def calcDesarme(self,x):
            
        electric_variables = [x[0],x[1],x[2],x[3],x[4],x[5]]
        #verificar a necessidade de converter eletric para DF

        
        input_desarme =  self.inputs_t + electric_variables
        #input_desarme = pd.concat[Inputs,eletrics_variables,axis = 1]

        cluster = kmeans.predict(input_desarme)

        Desarme = []

        for i in range(2,9): #indicar outputs desarme matriz modelo

            Desarme.append(Models[i][cluster[0]].predict(input_desarme))
       
            return Desarme


    def objective(self, x):
       
        obj =  calcConsumo(x)
        res_desarme = calcDesarme(x)
        eletrics_variables = [x(1),x(2),x(3),x(4),x(5),x(6)]

        desvio = [abs(res_desarme[0]- 54.5), abs(res_desarme[1]- 49.5),abs(res_desarme[2]- 850),abs(res_desarme[3]- 44),abs(res_desarme[4]- 43.5),abs(res_desarme[5]- 43.5),abs(res_desarme[6]- 43.5),abs(res_desarme[7]- 43.5),abs(res_desarme[8]- 44)]
        
        permitido = [0.05*54.5, 0.05*49.5, 0.05*850, 0.05*44, 0.05*43.5, 0.05*43.5, 0.05*43.5, 0.05*43.5, 0.05*44]

        for i in range(len(permitido)):
            if desvio[i] > permitido[i]:
                obj = 1e16

        for j in range(len(eletrics_variables)):

            if eletrics_variables[j] < lb[j] or eletrics_variables[j] > ub[j]:
                obj = 1e16

        return obj

    def minimizacao(self):

        sol = minimize(objective,x0,method='Nelder-Mead')
        return sol[0]

# =============================================================================