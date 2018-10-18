### [Tarefa com os datasets das eleições de 2006 e 2010](https://github.com/francinaldocn/AM/tree/master/Eleicoes/tarefa03_eleicoes2006_2010)

A tarefa pode ser encontrado no [link](https://github.com/francinaldocn/AM/tree/master/Eleicoes/tarefa03_eleicoes2006_2010)

Todos os modelos avaliados conseguiram estimar mais de 70% da variável indepentente (votos), no entanto, os modelos de regressão linear obtiveram um melhor desempenho na estimativa de votos, sendo o modelo com regressão linear com regularização LASSO o com maior *Coeficiete de determinação*, explicando 77,85% da variância dos votos estimadas pelos atributos selecionados para o modelo.

Dentre os modelos avaliados o que apresentou menor RMSE foi o Random Forest Regressor, acompanhados dos modelos de Regressão Linear com RIDGE, sem regularização e com regularização LASSO.


### [Tarefa com os datasets das eleições de 2006, 2010 e 2014](https://github.com/francinaldocn/AM/tree/master/Eleicoes/tarefa03_eleicoes2006_2010_2014)

A tarefa pode ser encontrado no [link](https://github.com/francinaldocn/AM/tree/master/Eleicoes/tarefa03_eleicoes2006_2010_2014)

Dos modelos avaliados nenhum conseguiu estimar mais de 65,18% da variável indepentente (votos). Assim como na [parte 01 dessa tarefa](https://github.com/francinaldocn/AM/tree/master/Eleicoes/tarefa03_eleicoes2006e2010) os modelos de regressão linear obtiveram um melhor desempenho na estimativa de votos, sendo o modelo com regressão linear com regularização RIDGE o com maior *Coeficiete de determinação*, explicando 65,17% da variância dos votos estimadas pelos atributos selecionados para o modelo. O modelo SVR foi o de menor desempenho 54,57%. 

Dentre os modelos avaliados o que deu melhor resultado em termos de RMSE foi a Regressão Linear com regularização RIDGE, no entanto, a regressão sem regularização e a com regularização LASSO tiveram valores de RMSE muito próximos da regressão com regularização RIDGE. 
