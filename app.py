import pandas as pd

from activation_functions import SignFunction
from perceptron import Perceptron



# Database dataset-treinamento:
    
dataset = pd.read_csv('database/dataset-treinamento.csv', sep=';', decimal=',')
X = dataset.iloc[:,0:3].values 
d = dataset.iloc[:,3:].values



perceptron = Perceptron(X, d, 0.01, SignFunction)  # entrada, saída, taxa de ativação e função de ativação
perceptron.train()


# Database dataset-teste:
    
dataset = pd.read_csv('database/dataset-teste.csv', sep=';', decimal=',')
X_teste = dataset.iloc[:,0:3].values 

for x in X_teste:
    y = perceptron.evaluate(x)
    print(f'Input: {x},Output: {y}')    