import os 
#os.environ["tf_gpu_allocator"]="cuda_malloc_async"


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import heapq
import math
import seaborn as sns


import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam



# Enable logging to see which CPU cores or operations are being placed
#tf.debugging.set_log_device_placement(True)


BS=200 #Batch SIze
LR=0.001 #Learning Rate
L1=100 #Number of units in Hidden Layer 1
L2=100 #Number of units in Hidden Layer 2
L3=100 #Number of units in Hidden Layer 3


per=0.01 #percentagem de VH




#Training Bit
file_path = "training_bit.txt"

with open(file_path, 'r') as myfile:
    data_training = [list(line.strip()) for line in myfile] # Read each line, convert to a list of characters

#Create the DataFrame
X_treinamento = pd.DataFrame(data_training).astype(int)
print(f"X_treinamento:\n{X_treinamento.describe()}\n")


#Validation Bit
file_path = "validation_bit.txt"

with open(file_path, 'r') as myfile:
    data_validation = [list(line.strip()) for line in myfile] # Read each line, convert to a list of characters

#Create the DataFrame
X_validacao = pd.DataFrame(data_validation).astype(int)
print(f"X_validacao:\n{X_validacao.describe()}\n")

#Testing Bit
file_path = "testing_bit.txt"

with open(file_path, 'r') as myfile:
    data_testing = [list(line.strip()) for line in myfile] # Read each line, convert to a list of characters

#Create the DataFrame
X_teste = pd.DataFrame(data_testing).astype(int)
print(f"X_teste:\n{X_teste.describe()}\n")




#Training Score
file_path = "training_score.txt"
y_treinamento_unravelled=pd.read_csv(file_path, header=None)
y_treinamento_unravelled.columns=['Score']
print(f"y_treinamento_unravelled:\n{y_treinamento_unravelled.describe()}\n")
y_treinamento=np.ravel(y_treinamento_unravelled)

y_treinamento_ar = heapq.nsmallest(math.floor(per*len(y_treinamento)), y_treinamento.T)
y_treinamento_lim = y_treinamento_ar[math.floor(per*len(y_treinamento))-1]
y_treinamento_VH = [x<=y_treinamento_lim for x in y_treinamento.T]

# Temos 2 classes
# Converter o vetor de rótulos para uma matriz binária
# Alterar o conjunto de dados de rótulos para o formato de string binária 1-de-N (one-hot encoding)
# Teremos uma rede neural com 2 neurônios na camada de saída
y_treinamento_VH = to_categorical(y_treinamento_VH)


#Validation Score
file_path = "validation_score.txt"
y_validacao_unravelled=pd.read_csv(file_path, header=None)
y_validacao_unravelled.columns=['Score']
print(f"y_validacao_unravelled:\n{y_validacao_unravelled.describe()}\n")
y_validacao=np.ravel(y_validacao_unravelled)

y_validacao_ar = heapq.nsmallest(math.floor(per*len(y_validacao)), y_validacao.T)
y_validacao_lim = y_validacao_ar[math.floor(per*len(y_validacao))-1]
y_validacao_VH = [x<=y_validacao_lim for x in y_validacao.T]

# Temos 2 classes
# Converter o vetor de rótulos para uma matriz binária
# Alterar o conjunto de dados de rótulos para o formato de string binária 1-de-N (one-hot encoding)
# Teremos uma rede neural com 2 neurônios na camada de saída
y_validacao_VH = to_categorical(y_validacao_VH)

#Testing Score
file_path = "testing_score.txt"
y_teste_unravelled=pd.read_csv(file_path, header=None)
y_teste_unravelled.columns=['Score']
print(f"y_teste_unravelled:\n{y_teste_unravelled.describe()}\n")
y_teste=np.ravel(y_teste_unravelled)

y_teste_ar = heapq.nsmallest(math.floor(per*len(y_teste)), y_teste.T)
y_teste_lim = y_teste_ar[math.floor(per*len(y_teste))-1]
y_teste_VH = [x<=y_teste_lim for x in y_teste.T]

# Temos 2 classes
# Converter o vetor de rótulos para uma matriz binária
# Alterar o conjunto de dados de rótulos para o formato de string binária 1-de-N (one-hot encoding)
# Teremos uma rede neural com 2 neurônios na camada de saída
y_teste_VH = to_categorical(y_teste_VH)



#num_objetos num_colunas
num_objetos, num_colunas = X_treinamento.shape


#Treino teste
#X_treino, X_teste, y_treino, y_teste = train_test_split(previsoes_df, previsoes_VH, test_size = 0.3, random_state = 42)
                                                                  
                                                                  
#Treinamento Validação
#X_treinamento, X_validacao, y_treinamento, y_validacao = train_test_split(X_treino, y_treino, test_size=0.2, random_state=42)


## Topologia da Rede Neural
# - relu:  função de ativação ReLU: f(x) = max(0, x)
# - softmax: cada neurônio de saída receberá uma probabilidade

modelo = Sequential()
modelo.add(Input(shape=(num_colunas,)))
modelo.add(Dense(units=L1,name='camada_escondida_1', activation='relu'))
modelo.add(Dense(units=L2,name='camada_escondida_2', activation='relu'))
modelo.add(BatchNormalization())
modelo.add(Dense(units=L3,name='camada_escondida_3', activation='relu'))
modelo.add(Dense(units=y_treinamento_VH.shape[1], name='camada_saida', activation='softmax')) #Ou é Virtual hit ou não é


## Parâmetros da Rede Neural
#  optimizer = algoritmo para atualizar os pesos baseado no loss:
#       Ex. 'adam', 'sgd', 'rmsprop'
#  loss = cálculo do erro.
#       Ex.: 'mean_squared_error','categorical_crossentropy', 'binary_crossentropy'
#  metrics = lista de métricas para avaliar o modelo

custom_adam = Adam(learning_rate=LR)  # Customizando para definir a learning_rate

modelo.compile(optimizer = custom_adam,
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])

modelo.summary()

#Treinamento da Rede Neural

# 150 épocas
# Atenção para o conjunto de validação.

modelo.fit(X_treinamento,
           y_treinamento_VH,
           batch_size = BS,
           epochs = 150,
           validation_data = (X_validacao, y_validacao_VH))


#Salvar modelo
sv_pth='BS_'+str(BS)+'_LR_'+str(LR)+'_'+str(L1)+'_'+str(L2)+'_'+str(L3)
modelo.save('./Resultados class/'+sv_pth+'/'+sv_pth+'.h5')


#Extração de previsões
previsoes = modelo.predict(X_teste)

# Como a RN está modelada com três saídas, precisamos buscar a posição da cadeia# de string binária que possui o maior valor (o valor True, i.e., 1)
# Tanto para a classe real quanto para a classe predita
y_previsao_matrix = [np.argmax(t) for t in previsoes]
y_teste_matrix = [np.argmax(t) for t in y_teste_VH]


# Acurácia do Modelo avaliada no Conjunto de Dados de teste:
accuracy = accuracy_score(y_teste_matrix, y_previsao_matrix)
print("Acurácia no conjunto de teste ( previsões Corretas / todas as previsões  ):\n", accuracy)



#Matriz de confusão - tabela com desempenho do modelo
confusao = confusion_matrix(y_teste_matrix, y_previsao_matrix)
print("Matriz de Confusão no Conjunto de Teste (Classes reais x Classes preditas):")

print("Matriz de Confusão")
print(confusao)

# Matriz de Confusão - representação mais informativa e didática
print("\n\n Matriz de Confusão")
sns.heatmap(
    confusao,
    annot=True,          # Show números nas células
    fmt='d',             # Números como inteiros
    cmap='Blues',        # Paleta de cores
    cbar=False,          # Remover barra à direita
    linewidths=1,        # Adicionar borda
    linecolor='black'    # Cor da borda
)
plt.xlabel('Classes preditas', fontsize=12)
plt.ylabel('Classes Reais', fontsize=12)
plt.title('Matriz de Confusão', fontsize=12)
plt.show()

#https://stackoverflow.com/questions/54141663/how-can-i-solve-error-allocation-exceeds-10-of-system-memory-on-keras

