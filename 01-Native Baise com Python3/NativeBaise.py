## Usando o Native Baise vamos fazer um programa que preveja a probabilidade de acidentes de veiculos

import pandas as pd ## Essa é a biblioteca que vamos usar para manipulação de dados 

from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
## Essa biblioteca permite fazermos a divisão entre os dados que serão utilizados para treino e para teste

from yellowbrick.classifier import ConfusionMatrix
## Essa biblioteca vai nos permitir visualizar a matriz de confusão 


baseDados = pd.read_csv('insurance.csv') 
## Nos estamos colocando dentro da váriavel (baseDados) os dados da planilha

baseDados = baseDados.drop(columns=['Unnamed: 0']) 
 
'''
Bom geralmente por ordenação do banco de dados geralmente se coloca um indice em cada linha da coluna para poder
manter um maior controle.
Essa planilha de exemplo que estamos usando também tem esses indices na primeira coluna (para o Python começa do 0)
sendo assim para evitar que esses números interfiram nos nossos calculos vamos apagar essa coluna referente ao indice

CUIDADO PARA NÃO APAGAR COLUNA ERRADA
''' 

baseDados.Accident.unique() ## Aqui estamos só visualizando a coluna aonde esta nossas classes (o codigo não funcinou)

'''
Nesse banco de dados de exemplo o NOME DA COLUNA AONDE ESTA NOSSAS CLASSES É "ACCIDENT". 
Nesta tabela não se adotou a convenção de colocar a classe como a ultima coluna da tabela, mas depois vamos
fazer a repartição 
'''

atributos = baseDados.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values
'''
Dentro das váriaveis atributos vamos guardar os atributos do nosso banco de dados, repare que temos todas as
colunas com os atributos EatributosCETO A COLUNA 7 que é justamente a coluna com a nossa classe. 

além disso note que no final usamos um (.values) para deixar esses dados no formato suportado pelo Native Base

Outra coisa que devemos observar é que o método que vai percorrer e retirar os elementos da variável para outra é o
(.iloc) a primeira os primeiros colchetes com 2 pontos [:] significa que queremos percorrer TODAS AS LINHAS da 
nossa tabela. O segundo colchete com os números de tabela representa QUAIS COLUNAS QUEREMOS PASSAR PARA NOSSA VÁRIAVEL

MAIS UMA VEZ REPARE QUE NÃO ESTAMOS PEGANDO A COLUNA REFERENTE A CLASSE 
'''

classe = baseDados.iloc[:,7].values

## Aqui é a mesma ideia criando uma váriavel para conter somente a coluna da nossa classe


####################### TRANSFORMAÇÃO DOS DADOS ####################################

'''
Se olharmos nossa tabela e até mesmo nossas váriaveis vamos ver que os dados guardados dentro dela estão no formato
literal (texto) entretanto o mais adequado, até mesmo por questão de processamento de dados, é utilizar valores
numericos representando cada um dos elementos dos atributos atributos.

Para fazer essa conversão podemos usar a função LabelEncoder() das bibliotecas que baixamos, mas para que possamos
fazer em cada um dos valores de cada atributo devemos percorrer TODOS OS ATRIBUTOS.  
'''

labelencoder = LabelEncoder() 
## Para não ter que ficar chamando a função toda hora vamos colocar a função LabelEncoder() dentro da váriavel
atributos[:,0] = labelencoder.fit_transform(atributos[:,0])
atributos[:,1] = labelencoder.fit_transform(atributos[:,1])
atributos[:,2] = labelencoder.fit_transform(atributos[:,2])
atributos[:,3] = labelencoder.fit_transform(atributos[:,3])
atributos[:,4] = labelencoder.fit_transform(atributos[:,4])
atributos[:,5] = labelencoder.fit_transform(atributos[:,5])
atributos[:,6] = labelencoder.fit_transform(atributos[:,6])
atributos[:,7] = labelencoder.fit_transform(atributos[:,7])
atributos[:,8] = labelencoder.fit_transform(atributos[:,8])
atributos[:,9] = labelencoder.fit_transform(atributos[:,9])
atributos[:,10] = labelencoder.fit_transform(atributos[:,10])
atributos[:,11] = labelencoder.fit_transform(atributos[:,11])
atributos[:,12] = labelencoder.fit_transform(atributos[:,12])
atributos[:,13] = labelencoder.fit_transform(atributos[:,13])
atributos[:,14] = labelencoder.fit_transform(atributos[:,14])
atributos[:,15] = labelencoder.fit_transform(atributos[:,15])
atributos[:,16] = labelencoder.fit_transform(atributos[:,16])
atributos[:,17] = labelencoder.fit_transform(atributos[:,17])
atributos[:,18] = labelencoder.fit_transform(atributos[:,18])
atributos[:,19] = labelencoder.fit_transform(atributos[:,19])
atributos[:,20] = labelencoder.fit_transform(atributos[:,20])
atributos[:,21] = labelencoder.fit_transform(atributos[:,21])
atributos[:,22] = labelencoder.fit_transform(atributos[:,22])
atributos[:,23] = labelencoder.fit_transform(atributos[:,23])
atributos[:,24] = labelencoder.fit_transform(atributos[:,24])
atributos[:,25] = labelencoder.fit_transform(atributos[:,25])

'''
Aqui estamos justamente fazendo a TRANSFORMAÇÃO de cada atributo percorrendo toda a linha dele
poderiamos usar um laço for, mas por questões didaticas foi feito dessa forma manual
'''

################################ CRIANDO AS VÁRIAVEIS PARA TREINO E TESTE #############################


atributosTreinamentos, atributosTestes, classeTreinamento, classeTeste = train_test_split(atributos, classe,
                                                                                          test_size = 0.3,
                                                                                          random_state = 0)


## aqui criamos váriaveis para receber dados de trei e de teste usando dados tanto de atributos quanto de classe

## usamos a função (train_test_split) que tinhamos baixado na biblioteca anterior. 

## Note que essa função recebe como parametro na ordem os atributos e a classe 

## Além disso devemos passar como parametro o (test_size) que é a quantidade por cento que queremos para teste. 

## neste caso passamos 30% (0.3) ao passo que por consequencia os outros 70% (0.7) serão para treino

## o último parametro que devemos passar agora é o (random_state) para que definirmos quais dados usar

## neste caso como passamos o parametro (0) indica que queremos usar os mesmos dados. 

modelo = GaussianNB() 
## Neste caso estamos criando o modelo usando a formula gausean Native Baise dentro da váriavel (modelo)

modelo.fit(atributosTreinamentos, classeTreinamento)

'''
Agora usamos a váriavel modelo (aonde está a native baise) para gerar o modelo usando os dados de treinamento..
No Python não temos como ver os modelos gerados (como ocorre no weka)

perceba que para gerar o modelo usamos o método (.fit) 
'''
previsoes = modelo.predict(atributosTestes)

'''
Agora nos criamos uma váriavel para, usando o modelo gerado anteriormente, fazermos os testes com os dados 
do (atributoTeste).

Perceba que para gerar os testes usamos o método (.predict)

Com as (previsoes) usando os atributos de Testes geramos previsoes que usando nossa I.A. 
Podemos comparar as (previsoes) com as (classesTestes), pois ela vai ter as respostas corretas assim podemos
já observar a porcentagem de acerto da nossa I.A 
'''



acuracidade = accuracy_score(classeTeste, previsoes)
'''
Usando a função (accuracy_score) passando como párametro a classeTeste e a nossá váriavel de previsões podemos
gerar o valor de porcentagem de acertos da nossa I.A

Neste exemplo nossa I.A acertou 0.8658 (86%)
'''



############################### MATRIZ DE CONFUSÃO ##################################

'''
Por meio da nossa biblioteca (ConfusionMatrix) podemos gerar a matriz de confusão em Python mostrando assim
de forma mais clara como foi o percentual de acerto da nossa I.A
'''

confusao = ConfusionMatrix(modelo, classes= ["Nenhum", "Severo", "Leve", "Moderado"] )
confusao.fit(atributosTreinamentos, classeTreinamento)
confusao.score(atributosTestes,classeTeste)
confusao.poof()

confusao = ConfusionMatrix(modelo, classes= ["None", "Severe", "Mild", "Moderate"] )
confusao.fit(atributosTreinamentos, classeTreinamento)
confusao.score(atributosTestes,classeTeste)
confusao.poof()