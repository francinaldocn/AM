
# Implementando Regressão Múltipla do Zero

Nessa tarefa você vai estender sua implementação da tarefa passada para considerar múltiplas variáveis. Você pode estender a versão vetorizada implementada [neste notebook](https://canvas.instructure.com/courses/1389733/files/68372962/download?verifier=jx3MnZn2ltc955iEt69PwjkzJvBiuQyUNXwGA9gl&wrap=1) para regressão simples. 

- Rode o algoritmo [nesses dados](https://canvas.instructure.com/courses/1389733/files/68372968/download?verifier=9Qm2NF62mBNlbfeLJWfrzjGl6qvS0eIuX3kOGNAU&wrap=1), onde as linhas representam as notas de alunos de computação de alunos da UFCG em algumas disciplinas do primeiro período. A última coluna é a variável alvo representando o CRA final depois de concluir o curso. As outras colunas são algumas disciplinas do primeiro período. O pressuposto aqui é que as notas em disciplinas no primeiro período ajudam a explicar o CRA final dos alunos de computação.

- Compare o valor dos coeficientes estimados pelo seu algoritmo com o valor dos coeficientes da regressão linear do scikit learn para testar se o seu algoritmo está funcionando corretamente.

A entrega deve ser o link no seu github para o notebook Jupyter com código python e texto explicativo quando necessário. De preferência, crie um repositório na sua conta do github e envie o link do html do notebook.

## Versão vetorizada para Regressão Linear Simples

A versão vetorizada de regressão linear simples servirá de base para esternder a implementação de uma versão vetorizada da regressão linear múltipla.

```
def compute_mse_vectorized(w,X,Y):
    res = Y - np.dot(X,w)
    totalError = np.dot(res.T,res)
    return totalError / float(len(Y))
    
def step_gradient_vectorized(w_current,X,Y,learningRate):
    res = Y - np.dot(X,w_current)
    b_gradient = np.sum(res)
    X = X[:,1][:,np.newaxis]
    m_gradient = np.sum(np.multiply(res,X))
    new_w = np.array([(w_current[0] + (2 * learningRate * b_gradient)),
             (w_current[1] + (2 * learningRate * m_gradient))])
    return [new_w,b_gradient,m_gradient]

def gradient_descent_runner_vectorized(starting_w, X,Y, learning_rate, epsilon):
    w = starting_w
    grad = np.array([np.inf,np.inf])
    i = 0
    while (np.linalg.norm(grad)>=epsilon):
        w,b_gradient,m_gradient = step_gradient_vectorized(w, X, Y, learning_rate)
        grad = np.array([b_gradient,m_gradient])
        #print(np.linalg.norm(grad))
        if i % 1000 == 0:
            print("MSE na iteração {0} é de {1}".format(i,compute_mse_vectorized(w, X, Y)))
        i+= 1
    return w

points = np.genfromtxt("income.csv", delimiter=",")
points = np.c_[np.ones(len(points)),points]
X = points[:,[0,1]]
Y = points[:,2][:,np.newaxis]
init_w = np.zeros((2,1))
learning_rate = 0.0001
#num_iterations = 10000
epsilon = 0.5
print("Starting gradient descent at w0 = {0}, w1 = {1}, error = {2}".format(init_w[0], init_w[1], compute_mse_vectorized(init_w, X,Y)))
print("Running...")
tic = time.time()
w = gradient_descent_runner_vectorized(init_w, X,Y, learning_rate, epsilon)
toc = time.time()
print("Gradiente descendente convergiu com w0 = {0}, w1 = {1}, error = {2}".format(w[0], w[1], compute_mse_vectorized(w,X,Y)))
print("Versão vetorizada rodou em: " + str(1000*(toc-tic)) + " ms")

```

## Versão vetorizada, estendida para Regressão Linear Múltipla (RLMV)


```python
import numpy as np
import math
import time
```


```python
def compute_mse_vectorized(w,X,Y):
    res = Y - np.dot(X,w)
    totalError = np.dot(res.T,res)
    return (totalError / float(len(Y)))
```


```python
def step_gradient_vectorized(w_current, X, y, learning_rate):
    m = len(y)
    res = Y - np.dot(X, w_current)
    gradient = -2 * (X.T @ res)
    new_w = w_current - (learning_rate/m) * X.T @ (X @ w_current - y)
    return [new_w, gradient]

```


```python
def gradient_descent_runner_vectorized(starting_w, X, y, learning_rate, epsilon, iterations):
    w = starting_w
    grad = np.linalg.norm(X)
    iter = 1
    #while (np.linalg.norm(grad) >= epsilon):
    for iter in range(iterations):
        w, gradient = step_gradient_vectorized(w,X,Y,learning_rate)
        grad = gradient
        if iter % 100000 == 0:
            print("MSE na iteração {0} é de {1}".format(iter,compute_mse_vectorized(w, X, Y)))
            iter+= 1
    return w
```

- Rode o algoritmo [nesses dados](https://canvas.instructure.com/courses/1389733/files/68372968/download?verifier=9Qm2NF62mBNlbfeLJWfrzjGl6qvS0eIuX3kOGNAU&wrap=1), onde as linhas representam as notas de alunos de computação de alunos da UFCG em algumas disciplinas do primeiro período. A última coluna é a variável alvo representando o CRA final depois de concluir o curso. As outras colunas são algumas disciplinas do primeiro período. O pressuposto aqui é que as notas em disciplinas no primeiro período ajudam a explicar o CRA final dos alunos de computação.


```python
# Carregar o dataset
points = np.genfromtxt("sample_treino.csv", delimiter=",")
# Remover a primeira linha da matriz que contém o nome das colunas
points = np.delete(points, 0,0)
# Adicionar uma coluna extra de 1, para w0
points = np.c_[np.ones(len(points)),points]
# Selecionando os atributos no dataset (variáres explicativas/independentes)
X = points[:, :-1]
# Selecionando a variável explicada/dependente
Y = points[:,-1][:,np.newaxis]
# Iniciando w com zeros
init_w = np.zeros(((len(points[0,:])-1),1))
# Taxa de aprendizado
learning_rate = 0.003 
# Número de iterações
iterations = 1000000
# 
epsilon = 1.7801187720365064e-09
print("Starting gradient descent at w0 = {0}, w1 = {1},  w2 = {2},  w3 = {3},  w4 = {4},  w5 = {5}, error = {6}".format(init_w[0], init_w[1], init_w[2], init_w[3], init_w[4], init_w[5], compute_mse_vectorized(init_w, X,Y)))
print("\nRunning...")
tic = time.time()
w = gradient_descent_runner_vectorized(init_w, X, Y, learning_rate, epsilon, iterations)
toc = time.time()
print("\nO gradiente descendente convergiu com w0 = {0}, w1 = {1}, w2 = {2},  w3 = {3},  w4 = {4},  w5 = {5}, error = {6}".format(w[0], w[1], w[2], w[3], w[4], w[5], compute_mse_vectorized(w,X,Y)))
print("\nA versão vetorizada rodou em: " + str(1000*(toc-tic)) + " ms")
```

    Starting gradient descent at w0 = [0.], w1 = [0.],  w2 = [0.],  w3 = [0.],  w4 = [0.],  w5 = [0.], error = [[54.47995386]]
    
    Running...
    gradient : 1.780077889634732e-09
    
    O gradiente descendente convergiu com w0 = [1.73771151], w1 = [0.10304143], w2 = [0.0464367],  w3 = [0.16409834],  w4 = [0.38117843],  w5 = [0.02027816], error = [[0.41133759]]
    
    A versão vetorizada rodou em: 20593.058586120605 ms


## Implementação da Regressão Linear Múltipa utilizando a biblioteca scikit-learn

A implementação utilizando o scikit será feita apenas para obenção dos coeficientes com o intuido de compará-los com os obtidos na implementação da Regressão Liner Múltipla vetorizada.

Para esse fim serão utilizadas apenas as bibliotecas numpy e sklearn


```python
### Importando as biblioteca numpy e sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
```


```python
# Dataset fornecido no notebook
points = np.genfromtxt('sample_treino.csv', delimiter=',')
```


```python
# Atributos - Variáveis independentes
X = points[1:, :-1]
# Objetivo - Variável dependente
y = points[1:,-1]
```


```python
# Ordinary least squares Linear Regression.
lm = LinearRegression()
```


```python
# Fit linear model
lm.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
# Mostrando os coeficientes estimados
print('Coeficientes Estimados')
print('w0 : {:.8f}'.format(lm.intercept_))
atributo = np.array(['w1','w2','w3','w4','w5'])
for i in range(0,5):
    print(atributo[i],": {:.8f}".format(lm.coef_[i]))
```

    Coeficientes Estimados
    w0 : 1.73771151
    w1 : 0.10304143
    w2 : 0.04643670
    w3 : 0.16409834
    w4 : 0.38117843
    w5 : 0.02027816


### Comparando o valor dos coeficientes estimados pelo algoritmo implementando (RMLV) e SCIKIT

 Tabela 01 - Comparativo dos coeficientes obtidos nas duas abordagens, RMLV e SCIKIT
 
    |  Coef  |    RLMV    |   SCIKIT   |
    |--------|------------|------------|
    |   w0   | 1.73771151 | 1.73771151 |
    |   w1   | 0.10304143 | 0.10304143 |
    |   w2   | 0.0464367  | 0.04643670 |
    |   w3   | 0.16409834 | 0.16409834 |
    |   w4   | 0.38117843 | 0.38117842 |
    |   w5   | 0.02027816 | 0.02027815 |
    
Pela Tabela 01, verificamos que os coeficientes obtidos da implementação da regressão linear múltipla vetorizada - RLMV, obtivemos os mesmos valores do scikit learn. 

Desse modo, verifica-se que a o algoritmo implementado está funcionando corretamente.


### Modelo

 $\hat{y}$ = 1.73771151 + 0.10304143$\cdot$**Cálculo1** + 0.04643670$\cdot$**LPT** + 0.16409834$\cdot$**P1** + 0.38117843$\cdot$**IC** + 0.02027816$\cdot$**Cálculo2** + $\epsilon$

### Interpretando os coeficientes:

- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **Cálculo1** está associado a um **aumento de 0.10304143** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **LPT** está associada a um **aumento de 0.04643670** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **P1** está associada a um **aumento de 0.16409834** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **IC** está associada a um **aumento de 0.38117843** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **Cálculo2** está associado a um **aumento de 0.02027816** no CRA.
