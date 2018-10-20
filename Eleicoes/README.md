
# Previsão de Votação de Deputados

### Sobre a tarefa

Nesta atividade construiremos modelos preditivos de regressão em scikit learn para a predição dos votos de deputados federais considerando as últimas eleições. As atividades esperadas para essa etapa são descritas a seguir:

1. Baixe os dados [aqui](https://canvas.instructure.com/courses/1389733/files/69523670/download?verifier=A5EPvssqIQCjlxpWQyesLFer1VDTlRyTAAXR2iyi&wrap=1) (veja descrição dos dados mais abaixo).
2. Considere o pipeline mostrado nesse [link](https://www.kaggle.com/apapiu/regularized-linear-models) para construir seus modelos de regressão. Isso implica, dentre outras coisas:
    - Analisar as distribuições das variáveis para ver se estão enviesadas e precisam de correção; tratamento de valores ausentes, variáveis categóricas e normalização, quando for o caso.   
    - Construir modelos de regressão com (ridge e lasso) e sem regularização.    
    - Considerar também modelos de regressão não paramétrica como K-NN.    
    - Considerar outros modelos ainda não vistos em sala de sua escolha (e.g. SVR, Regression Trees e Random Florests).    
    - Tunar os hiperâmetros para cada caso e retornar os rmses de validação cruzada para todos os modelos avaliados.    
    - Plotar os resíduos versus predições e analisar se esses plots representam bons indícios da adequabilidade dos modelos a esse problema.    
3. Alguns dias antes da entrega final serão liberados os dados de teste referentes à 2014 para validação final dos seus melhores modelos.
    - Dica: Uma coisa que você pode fazer é usar os dados de 2006 como treino e os de 2010 como validação. Uma vez encontrados os melhores modelos para 2010 junte 2006+2010, retreine, e aplique o modelo aos dados de 2014 que serão liberados.    
4. Responder:
    - Dentre os modelos avaliados, qual foi o que deu o melhor resultado nos dados de 2014 em termos de RMSE? Justifique bem sua resposta.
    
A entrega deve ser um notebook Jupyter com código python e texto explicativo quando necessário. Crie um repositório na sua conta do github e envie o link do html do notebook.

### Sobre os dados

Vamos explorar dados sobre as votações que candidatos à Câmara Federal de Deputados receberam nos anos de 2006 e 2010. Esses dados foram extraídos do [TSE](http://www.tse.jus.br/hotSites/pesquisas-eleitorais/index.html), pré-processados e contemplam informações sobre aproximadamente 7.300 candidatos. A descrição de cada atributo é dada mais abaixo.

|  Variável  | Tipo | Descrição |
|:------------|:------:|:-----------|
|"sequencial_candidato" | (character) | id do candidato|
|"nome"| (character) | |
|"uf" | (character) | |
|"partido" | (character)| |
|"quantidade_doacoes"| (integer)| |
|"quantidade_doadores"| (integer) | número de doadores diferentes|
|"total_receita" | (double) | soma em R\$ das doações |
|"media_receita" | (double) | média das doações |
|"recursos_de_outros_candidatos/comites" | (double) | quantia em R\$ das doações provenientes de outros candidatos ou comite partidário |
|"recursos_de_pessoas_fisicas" | (double) | quantia em R\$ das doações provenientes de outros CPFs |
|"recursos_de_pessoas_juridicas" | (double) | quantia em R\$ das doações provenientes de outros CNPJ |
|"recursos_proprios" | (double) | quantia em R\$ das doações provenientes do próprio candidato |
|"recursos_de_partido_politico" | (double) | quantia em R\$ das doações provenientes do partido político do candidato |
|**"votos"** | (integer) | **variável alvo**. Se refere ao número de votos na campanha de 2006 e 2010 |
|"quantidade_despesas" | (integer) | |
|"quantidade_fornecedores" | (integer) | número de fornecedores/despesas diferentes |
|"total_despesa" | (double) | soma em R$ das despesas de campanha |
|"media_despesa" | (double) | média das despesas de campanha |
|"cargo" |(character) | |
|"Sexo": | (character) | |
|"grau": | (character) | grau de instrução do candidato |
|"estado_civil" | (character) |
|"ocupacao" | (character) | ocupação do candidato |



### [Tarefa com os datasets das eleições de 2006 e 2010](https://github.com/francinaldocn/AM/blob/master/Eleicoes/tarefa03_eleicoes_2006_2010.ipynb)

O notebook da tarefa pode ser encontrado no [link](https://github.com/francinaldocn/AM/blob/master/Eleicoes/tarefa03_eleicoes_2006_2010.ipynb)

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MODELO</th>
      <th>RMSE</th>
      <th>R-QUADRADO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Random Forest Regressor</td>
      <td>27338.0</td>
      <td>0.772915</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN Regressor</td>
      <td>28760.0</td>
      <td>0.742536</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Regression trees</td>
      <td>28929.0</td>
      <td>0.740210</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Regressão Linear com RIDGE</td>
      <td>32142.0</td>
      <td>0.768698</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Regressão Linear com LASSO</td>
      <td>32168.0</td>
      <td>0.768794</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Regressão Linear sem regularização</td>
      <td>32950.0</td>
      <td>0.768325</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SVR</td>
      <td>35192.0</td>
      <td>0.703907</td>
    </tr>
  </tbody>
</table>
</div>



Dos modelos avaliados, aplicados nos dados referente à votação dos deputados nas eleições de 2006 e 2010, chegamos as seguintes conclusões:

 - Todos os modelos avaliados conseguiram estimar mais de 70% da variável alvo (votos).
 - O modelo **Random Forest Regressor** obteve o melhor desempenho, pois apresentou **menor RMSE** e **maior R²** que os demais modelos.
 - Os modelos com regressão linear tiveram um desempenho muito próximo, sendo o modelo com *regularização RIDGE* o que apresentou o *menor RMSE*.
 - Dentre os modelos avaliados o que apresentou menor desempenhro foi o *SVR*, com *maior RMSE* e *menor R²*.


### [Tarefa com os datasets das eleições de 2006, 2010 e 2014](https://github.com/francinaldocn/AM/blob/master/Eleicoes/tarefa03_eleicoes_2006_2010_2014.ipynb)

O notebook da tarefa pode ser encontrado no [link](https://github.com/francinaldocn/AM/blob/master/Eleicoes/tarefa03_eleicoes_2006_2010_2014.ipynb)

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
      <th>MODELO</th>
      <th>RMSE</th>
      <th>R-QUADRADO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Random Forest Regressor</td>
      <td>32997.0</td>
      <td>0.645016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN Regressor</td>
      <td>34123.0</td>
      <td>0.616291</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Regression trees</td>
      <td>34329.0</td>
      <td>0.620255</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Regressão Linear com LASSO</td>
      <td>36281.0</td>
      <td>0.650532</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Regressão Linear com RIDGE</td>
      <td>36409.0</td>
      <td>0.651481</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Regressão Linear sem regularização</td>
      <td>36909.0</td>
      <td>0.651464</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SVR</td>
      <td>39019.0</td>
      <td>0.545786</td>
    </tr>
  </tbody>
</table>
</div>



Dos modelos avaliados, aplicados nos dados referente à votação dos deputados nas eleições de 2006, 2010 e 2014, chegamos as seguintes conclusões:

 - Todos os modelos avaliados conseguiram estimar no máximo 65% da variável alvo (votos).
 - O modelo **Random Forest Regressor** apresentou **menor RMSE** que os demais modelos.
 - Os modelos com regressão linear tiveram um desempenho muito próximo, sendo o modelo com *regularização LASSO* o que apresentou o *menor RMSE*.
 - Dentre os modelos avaliados o que apresentou menor desempenhro foi o *SVR*, com *maior RMSE* e *menor R²*.



Comparando as duas abordagens, a [primeira](https://github.com/francinaldocn/AM/blob/master/Eleicoes/tarefa03_eleicoes_2006_2010.ipynb) como modelos gerados com os dados de 2006 e 2010 e a [segunda](https://github.com/francinaldocn/AM/blob/master/Eleicoes/tarefa03_eleicoes_2006_2010_2014.ipynb) utilizando os dados de 2006, 2010 e 2014, percebemos que, mesmo utilizando a mesma estrutura, a estimação dos modelos foi bem diferente. Enquanto na primeira abordagem conseguiu-se explicar até 77,29% da variância com o modelo *Random Forest Regressor* na segunda, conseguiu-se no máximo 65,14% com a *Regressão Linear com RIDGE*

Mesmo tendo uma alta variância nos dados, o que prejudica a construção dos modelos, o menor RMSE ainda foi maior que a média dos votos.
