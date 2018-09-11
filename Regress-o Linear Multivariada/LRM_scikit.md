
# Implementando Regressão Múltipla

## Implementação utilizando a biblioteca scikit-learn

### Com a biblioteca numpy


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
points = np.genfromtxt('sample_treino.csv', delimiter=',')
```


```python
# Arrays x e y
```


```python
# Parametros
X = points[1:, :-1]
y = points[1:,-1]
```


```python
from sklearn.linear_model import LinearRegression
```


```python
lm = LinearRegression()
```


```python
lm.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
### Avaliação do modelo
```


```python
print(lm.intercept_)
```

    1.737711513794439



```python
print('Coeficientes')
atributo = np.array(['Cálculo1','LPT','P1','IC','Cálculo2'])
for i in range(0,5):
    print(atributo[i],":",round(lm.coef_[i],6))
```

    Coeficientes
    Cálculo1 : 0.103041
    LPT : 0.046437
    P1 : 0.164098
    IC : 0.381178
    Cálculo2 : 0.020278


### Forma da regressão linear multivariada para o dataset

 $\hat{y}$ = 1.737711513794439 + 0.103041$\cdot$**Cálculo1** + 0.046437$\cdot$**LPT** + 0.164098$\cdot$**P1** + 0.381178$\cdot$**IC** + 0.020278$\cdot$**Cálculo2**

Interpretando os coeficientes:

- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **Cálculo1** está associado a um **aumento de 0.103041** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **LPT** está associada a um **aumento de 0.046437** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **P1** está associada a um **aumento de 0.164098** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **IC** está associada a um **aumento de 0.381178** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **Cálculo2** está associado a um **aumento de 0.020278** no CRA.

### Com a biblioteca pandas


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
points2 = pd.read_csv('sample_treino.csv')
```


```python
points2.columns
```




    Index(['Cálculo1', 'LPT', 'P1', 'IC', 'Cálculo2', 'cra'], dtype='object')




```python
# Parametros
X1 = points2[['Cálculo1', 'LPT', 'P1', 'IC', 'Cálculo2']]
y1 = points2['cra']
```


```python
from sklearn.linear_model import LinearRegression
```


```python
lm2 = LinearRegression()
```


```python
lm2.fit(X1, y1)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
print(lm2.intercept_)
```

    1.7377115137944363



```python
coeff_points = pd.DataFrame(lm2.coef_,X1.columns,columns=['Coefficient'])
coeff_points
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cálculo1</th>
      <td>0.103041</td>
    </tr>
    <tr>
      <th>LPT</th>
      <td>0.046437</td>
    </tr>
    <tr>
      <th>P1</th>
      <td>0.164098</td>
    </tr>
    <tr>
      <th>IC</th>
      <td>0.381178</td>
    </tr>
    <tr>
      <th>Cálculo2</th>
      <td>0.020278</td>
    </tr>
  </tbody>
</table>
</div>



### Forma da regressão linear multivariada para o dataset

 $\hat{y}$ = 1.737711513794439 + 0.103041$\cdot$**Cálculo1** + 0.046437$\cdot$**LPT** + 0.164098$\cdot$**P1** + 0.381178$\cdot$**IC** + 0.020278$\cdot$**Cálculo2**

Interpretando os coeficientes:

- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **Cálculo1** está associado a um **aumento de 0.103041** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **LPT** está associada a um **aumento de 0.046437** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **P1** está associada a um **aumento de 0.164098** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **IC** está associada a um **aumento de 0.381178** no CRA.
- Mantendo todas as outras variáveis constantes, um aumento de 1 unidade em **Cálculo2** está associado a um **aumento de 0.020278** no CRA.
