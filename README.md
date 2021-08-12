# **PROJETO - PREVENÇÃO DE DANOS: Utilização de Machine Learning na predição de UTI para pacientes acometidos pelo COVID-19**

[<img src="https://img.shields.io/badge/author-Glaudemias-yellow?style=flat-square"/>](https://github.com/Glaudemias)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AUkgLzHrnZaVQEP6glJpNukolC79vGoR?authuser=1#scrollTo=SGJnfmkjPhUz)


<img src="Images/IMAGES/capa_covid (2).jpg"/>

# Conteúdo 
1. [INTRODUÇÃO](#in)
  * Apresentação
  * COVID-19
  * Machine Learning
  * Machine Learning aplicado à Saúde
2. [DADOS](#dt)
  * Entendendo os Dados
  * Análise Exploratória dos Dados
3. [METODOLOGIA](#mt)
  * Ferramentas
  * Machine Learning
4. [ESTRUTURA DO REPOSITÓRIO](#es)
  * Data
  * Images
  * Notebooks
5. [CONCLUSÕES](#concl)

6. [AGRADECIMENTOS](#ag)

7. [REFERÊNCIAS BIBLIOGRÁFICAS](#rb)

<a name="in"></a>
## INTRODUÇÃO
### Apresentação
 Olá! me chamo Glaudemias Grangeiro Júnior. Sou um nordestino-cearense, arquiteto e urbanista e um apaixonado por estudos de cidades. Tenho fascinio por dados desde a graduação, quando trabalhei diretamente com dados geoespaciais e georreferenciamento.E há mais ou menos um ano e meio resolvi entrar no mundo de Ciências de Dados

Aqui você encontrará o ultimo projeto (de um total de 4), realizados ao longo do Bootcamp DataScience Aplicada ofertado pela Alura. O projeto em questão assim como o [anterior](https://github.com/Glaudemias/Projeto_COVID_Sertao_CENTRAL) irá tratar da pandemia de COVID-19, especificamente no Brasil, utilizando a base de dados do Sírio Libanes disponibilizado no [Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19) e dados do portal do governo federal sobre os dados de [COVID-19](https://covid.saude.gov.br/).

Esse projeto a fim de ter uma melhor organização e compreensão por você (e qualquer outra pessoa que vá analisa-lo) foi dividido em 3 partes. Na seção de Metodologia abaixo isso será melhor explanado.


Enfim, espero que gostem 😄

### COVID-19

Desde quando foi noticiado o primeiro caso de pessoa infectada pelo vírus coronavírus (SARS-COV 2), grandes têm sido os esforços da comunidade científica e  de alguns líderes mundiais, na contenção e achatamento da propagação da doença. Por ser um vírus de disseminação comunitária, medidas como o isolamento social e distanciamento social foram levantadas no dia 25 de Fevereiro de 2020 pelo relatório de pesquisadores chineses, mostrando que as medidas de quarentena adotadas pelo país havia sido  o primeiro a ser contaminado pela pandemia e o primeiro a tomar medidas de restrição[1] 

No Brasil o  primeiro caso de covid-19 foi registrado em 06/02/2020, na cidade de São Paulo. Em 30 de janeiro de 2020, a Organização Mundial da Saúde (OMS) declarou a epidemia emergência internacional, o mundo teve que se reorganizar e tentar conviver com uma nova realidade marcada por incertezas, mortes, depositando esperança na criação da vacina. Hoje com a vacinação em andamento desde 08 de dezembro de 2020 (primeira vacinada no mundo), já se vislumbra uma possível saída de todo esse caos.[2] 

Contudo a pandemia não está nada perto de passar, as medidas de contenção como o isolamento social são importantes para o achatamento da curva de contaminação e consequentemente evitar a crise hospitalar, causada pela demanda superior a capacidade do sistema de saúde. Sendo assim, preparar nosso sistema de saúde para melhor prover um atendimento e demanda por leitos e atendimentos, é essencial. E nessa perspectiva dados individualizados baseados em atendimentos já feitos podem auxiliar na demanda de leitos de Unidade Intensiva de Tratamento, EPI e profissionais disponíveis para um atendimento seguro e eficaz.      

### Machine Learning

Machine Learning é uma área da ciência e engenharia da computação, que trata da capacidade automatizada de se ensinar uma máquina, esse aprendizado pode ser supervisionado ou não[3]. Através da elaboração de algoritmos é possível coletar dados, aprender com eles e então fazer uma predição, determinação sobre alguma coisa no mundo [4]. As aplicações de Machine Learning são as mais variadas, desde ler e identificar um e-mail como spam ou não, até visão computacional e como dito anteriormente o tipo de aprendizagem dessa máquina se divide em dois tipos principais, aqui serão definidos com base no escrito de [5] e [6]:
* **Aprendizado não  supervisionado**:   
Quando falamos de um aprendizado de Máquina não supervisionado, não temos um resultado esperado como na elaboração de um modelo que deseja saber a melhor rota de um carro, nesse caso não existe um resultado esperado ou resposta certa. O aprendizado não supervisionado nos permite obter um agrupamento de conjuntos, mesmo sem estarem categorizados, por outro lado, por ser subjetivo, torna-se difícil definir métricas de quão boa está a performance do modelo.
A técnica de agrupamento é chamada **clustering** sua vantagem se dá pelo fato de a máquina não ficar limitada aos rótulos dados pelo dataset, então é capaz de formar grupos por si só, levando em consideração as características mais importantes
* **Aprendizado supervisionado**:
Em problemas de aprendizagem supervisionada é necessário que haja um conjunto de dados rotulados, que representam o conjunto de dados de características, por exemplo se o conjunto de dados é sobre características de um animal (tamanho dos olhos, cor do pelo, peso e tamanho), é necessário também rotular o conjunto dessas características, para que o algoritmo a ser utilizado faça a relações entre esses dados, sendo capaz de classificar gatinhos que não estejam rotulados. Os problemas de classificação são divididos em dois tipos: **Regressão** e **Classificação**, para explicar cada um, é necessário saber o conceito de tipos de rótulos, podem ser: **Numérica Contínuo**(datas, tamanhos e comprimentos) ou **Categórica**(gênero, tipo de material e método de pagamento).

**Classificação** A classificação utiliza o rótulo do tipo dados  categóricos, já que seu objetivo é determinar a classe a que o conjunto pertence. A **Regressão** utiliza o rótulo do tipo numérico, que o permite estimar o valor baseado nas características do elemento, logo a saída desse modelo é também do tipo numérico contínuo.

### Machine Learning aplicado à Saúde

Entendendo que o potencial do aprendizado de máquina está diretamente ligado com a adição de dados a um modelo, não demorou para que se utilizassem os benefícios providos por essa tecnologia  nas diversas áreas da sociedade, dentre elas a saúde.Por meio de equipamentos e algoritmos que se baseiam em Machine Learning, é possível cruzar os dados clínicos dos pacientes com as referências epidemiológicas obtidas nos sites governamentais em saúde e formular diversas hipóteses sobre um evento clínico. Dessa forma o uso de Machine Learning na saúde fornecerá conclusões mais apuradas do que se fossem analisadas isoladamente, trazendo mais segurança ao profissional de saúde quando tomar as decisões pelo paciente[7].

De acordo com [8] as aplicações possíveis para a utilização de Machine Learning na área da saúde são:

* **Direcionamento:** Uma das preocupações decorrentes do novo coronavírus é o isolamento de pessoas sintomáticas para reduzir o contágio e propagação da doença. Existem processos inteligentes capazes de relacionar sintomas apresentados com a doença, resultando em direcionamento mais ágil dos infectados;
* **Diagnostico de Câncer:** Ao analisar as características de um nódulo na imagem de ressonância, segundo um banco de dados já definido, é possível classificar o nódulo como maligno ou benigno. Também é possível saber a probabilidade de o nódulo pertencer a cada uma das classificações, para então definir o diagnóstico do paciente e os próximos passos do tratamento;
* **Triagem:** O uso de machine learning na saúde também pode não ser tão específico, indicando não apenas uma, mas as possíveis enfermidades que o paciente pode ter de acordo com os sintomas que ele apresenta. Uma aplicação possível seria durante a triagem: ao chegar no pronto atendimento, diante dos sintomas apresentados e com o auxílio dos dados já pré-definidos no sistema, o paciente é encaminhado ao especialista mais adequado;
* **Entendimento do contexto:** aplicativos de pesquisa podem contribuir para a compreensão do avanço de pandemias, como a da COVID-19, por meio de testes não invasivos. Por exemplo, as pessoas respondem sobre seu quadro clínico atual e, com o passar do tempo, é possível avaliar o avanço da doença na região. Neste sentido, é possível agir preventivamente quanto à disponibilidade de equipamentos e profissionais para atender às diferentes demandas regionais.

Do ponto de vista político administrativo a ciência de dados e o Machine Learning, estão se mostrando  ferramentas essenciais para lidar com a pandemia de Coronavírus, já que as ferramentas permitem um maior monitoramento, leitura da realidade e possibilidade de predição dos números de casos. Mas do ponto de vista do tratamento de pacientes acometido pela síndrome respiratória causada pelo novo Coronavírus as técnicas de Machine Learning, podem auxiliar na predição de leitos de hospitais, atendimento por equipe médica, EPI e tudo o que for necessário para o tratamento da doença, a partir das experiências passadas e das base de dados construída por hospitais ao longo do tempo [7].

<a name="dt"></a>
## **DADOS**
### Entendendo os Dados
Os dados desse notebook foram retirados de dois locais:
* **A primeira base** de dados é da plataforma **[Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19)** e são referentes aos dados de internação do hospital Sírio Libanês;
* **A segunda base, são dos dados referentes aos casos e óbitos de Covid** e foram óbtidos pelo **portal oficial do [governo federal](https://covid.saude.gov.br/)**. A segunda base em questão foi a mesma utilizada no projeto anterior, cujo tema também abordava a COVID-19 no Brasil, especificamente no Estado do Ceará e você pode encontrar mais informações sobre a base de dados [aqui](https://github.com/Glaudemias/Projeto_COVID_Sertao_CENTRAL).

Aqui iremos detalhar especificamente a base de dados do Kaggle, buscando enteder sua composição e as suas features mais importantes:

A base de dados em questão é feita por dados anonimizados como forma de garantir a ética de cada paciente, mas em explorações no repositório do kaggle, é possível encontrar algumas informações sobre esses valores binários. Dessa forma podemos entender a base de dados composta de:

* **1925** linhas e **231** colunas 

Dentre elas num primeiro momento podemos destacar três principais colunas são elas:
 * **ICU**:  Possui como valores 0 e 1 e segundo a base do Kaggle, representam os valores 0 Não irá para UTI e 1 irá para UTI; 

Exemplificando:


|ICU|DESCRIÇÃO|
|:-------:|:------------------------:|
| 0	      | Não deu entrada na UTI   |
| 1	      | Deu entrada na UTI       |


* **WINDOW**: Essa coluna trata da informação dos pacientes que deram ou não entrada na UTI, ela é dividida em 5 valores, relacionados ao tempo que um paciente levou para conseguir da entrada na UTI são eles:

* *Pacientes que deram entrada na UTI nas primeiras 2 horas;*
* *Pacientes que deram entrada na UTI entre 2 horas e 4 horas;*
* *Pacientes que deram entrada na UTI entre 4 e 6 horas;*
* *Pacientes que deram entrada na UTI entre 6 e 12 horas;*
* *Pacientes que entraram na UTI após 12 horas.*

|WINDOW|DESCRIÇÃO|
|:-------:|:-----------------------------------:|
| 0-2	    | Paciente entrou entre 0 e 2 horas   |
| 2-4	    | Paciente entrou entre 2 e 4 horas   |
| 4-6	    | Paciente entrou entre 4 e 8 horas   |
| 6-12    |	Paciente entrou entre 8 e 12 horas  |
| Above-12| Paciente entrou após 12 horas       |

* **PATIENT_VISIT_IDENTIFIER**: Observando a coluna com mais atenção, percebe-se que os valores atribuidos as informações valores máximos `max` e minímos `min`, temos a seguinte condição:

* 0 como valor mínimo; 
* 384 como valor máximo;

Essa informação confirma que as 1925 linhas não são respecticas cada uma a um único paciente. Se olharmos o dataframe acima teremos o seguinte cenário:

|ID |PATIENT_VISIT_IDENTIFIER|DESCRIÇÃO|
|:-:|:--------:|-----------------------------------:|
| 0 |0	        | Paciente identificado como 0       |
| 1 |0         | Paciente identificado como 0       |
| 2 |0	        | Paciente identificado como 0       |
| 3 |0         |	Paciente identificado como 0       |
| 4 |0         | Paciente identificado como 0       |
| 5 |1	        | Paciente identificado como 1       |
| 6 |1         |	Paciente identificado como 1       |
| 7 |1         | Paciente identificado como 1       |
| 8 |1	        | Paciente identificado como 1       |
| 9 |1         |	Paciente identificado como 1       |

Em outras palavras, temos uma base de dados **desnormalizada** ja que possuimos um dataframe composto de várias linhas que identificam um único paciente. Não possuímos **1925 pacientes**, mas sim **385 pacientes**.

De maneira geral toda a base de dados se divide em **54 variáveis** composta de **4 principais grupos**:

* 1º **Informação demográfica** -**3** variáveis do tipo **categórica**;
* 2º **Doenças pré-existentes** - **9** variáveis do tipo **categórica**;
* 3º **Resultados de exame de sangue** - **36** variáveis do tipo **contínua**;
* 4º **Sinais vitais** - **6** variáveis do tipo **contínua**.

### TRATAMENTO DOS DADOS
Ao longo do projeto foram realizados os seguintes processos para o tratamento da base de dados:
* **Identificação e Tratamento dos valores vazios**;
* **Removeção de pacientes que chegaram e foram diretos para a UTI na janela 0-2**;
* **Utilização dos dados de janela das primeias duas horas**;
* **Dividizão da base de dados em dois dataframes: dados de análise e dados para Machine Learning**;
* **Categorização dos valores da coluna AGE_PERCENTIL**;
* **Eliminar as colunas que são altamente correlacionadas a firm de evitar erros no Machine Learning**.

### Analise exploratória dos Dados
Nessa etapa foram analisados as duas base de dados.

* Em um primeiro momento foram analisados os dados de COVID-19 contendo os casos e óbitos, com foco principalmente nos municípios de São Paulo - SP e Brasilia - DF, localidades onde estão implantados alguns centros de atendimento do Sírio Libanês. A partir dessa base de dados foram traçadas duas perguntas norteadoras que deram norte para a exploração da base de dados, são elas:

1. **Quais os municipios brasilieros que mais foram afetados pela pandemia de COVID-19, em numero de casos?**

2. **Quais os municipios brasilieros que mais foram afetados pela pandemia de COVID-19, em numero de óbitos?**

Essas perguntas possibilitaram que fossem elaboradas algumas formas de visualizações de dados, focadas em respondê-las. São elas:

 * Uma espacialização dos dados em **mapas interativos** (você pode baixa-lo [aqui](https://github.com/Glaudemias/PROJETO_FINAL/blob/main/Images/HTML/Mapas%20Interativos%20-%20Covid-19.rar) ou  observar a visualização da versão interativa do notebook [aqui](https://nbviewer.jupyter.org/github/Glaudemias/teste/blob/main/PROJETO_FINAL_parte_2_An%C3%A1lise_Explorat%C3%B3ria_da_base_de_Dados.ipynb))
 * Visualização do número de casos crescendo por meio de centroides, representados aqui por meio de **gifs**; 
 * Um Ranking dos 13 municipios mais afetados em número de **casos e óbitos** por Covid-19, feitos a partir de um gráfico de barras.

<img src="Images/GIFS/mapa (3) (2).gif" width="450"/>  <img src="Images/GIFS/mapa2 (1).gif" width="450"/> 

* Em um segundo momento foram analisados os dados de pacientes, disponibilizados pela base do Sírio Libanês. A partir dessa base de dados foram traçadas 7 perguntas norteadoras que deram norte para a exploração da base de dados, são elas:

1. **Qual genero que mais prevaleceu dentro da base de dados?** 
2. **Existe mais, ou menos pessoas acima de 65 anos?**
3. **Quais os percentis de idades com maior número de casos ?**
4. **Como estão distribuidos a relação de Gênero por Idade?**
5. **Como estão distribuidos a relação de Gênero por entrada na UTI?**
6. **Como estão distribuidos a relação de percentis de idade por entrada na UTI?**
7. **Como estão distribuidos a relação de grupos de doença por entrada na UTI?**

Essas perguntas possibilitaram que fossem elaboradas algumas formas de visualizações de dados, focadas em respondê-las. São elas:

* Análise do número de pacientes a partir da distribuição por gênero;
* Análise do número de pacientes a partir da distribuição por idade;
* Análise do número de pacientes a partir da relação entre gênero e idade;
* Análise do número de pacientes a partir da distribuição por gênero e sua relação com a entrada (ou não ) na UTI;
* Análise do número de pacientes a partir da distribuição por idade e sua relação com a entrada (ou não ) na UTI;
* Análise do número de pacientes a partir da distribuição por grupos de doenças e sua relação com a entrada (ou não ) na UTI.

<a name="mt"></a>
## **METODOLOGIA**

O presente projeto foi pensado a partir de uma estrutura básica, sendo ela exemplificada no diagrama abaixo:

<img src="Images/IMAGES/METODOLOGIA.jpg"/>

De forma resumida podemos entender esse diagrama da seguinte forma:

* 1: Corresponde a importação das bibliotecas utilizadas para manipulação, tratamento, visualização, leitura e processamento dos dados, além da elaboração dos modelos de Machine Learning;
* 2: A segunda etapa é o tratamento dos dados, retirando as colunas desnecessárias, tratando os valores vazios, reduzindo o número de informações e dividindo o dataset em uma parte destinada a um dataframe de análise exploratória dos dados e outro destinado a eleboração de modelos de Machine Learning;
* 3: A terceira etapa é a análise exploratória dos dados. Aqui afim de fazer uma leitura do problema da covid, foi utilizado os dados coletados na plataforma governamental do [COVID-19](https://covid.saude.gov.br/)(dados respectivos ao dia 25/07/2021) utilizados no projeto anterior desse bootcamp e disponivel nesse repositório do [Github](https://github.com/Glaudemias/Projeto_COVID_Sertao_CENTRAL) e também os dados disponibilizados pelo hospital Sírio Libanês, na sua plataforma do [Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19);  
* 4: A quarta etapa é a divisão dos dados em dados de treino e dados de teste, que serão utilizados para treinar e validar a performance dos modelos de Machine Learning elencados para esse projeto
* 5: A quinta etapa é o treinamento dos dados de teste com cada um dos 4 modelos elencados pra esse projeto.
* 6: A sexta etapa é a avaliação da performance de cada modelo, utilizando métricas como: AUC e Acurácia
* 7: A implementação e exportação do modelo de Machine Learning que teve melhor performance e as considerações sobre todo o processo.

Vale a pena salientar que toda essa estrutura foi dividia em 3 notebooks, como pode ser explicado abaixo no tópico de [estrutura](#es).

### FERRAMENTAS
Esse projeto foi feito em python 3.8.5, usando o google Colab como IDE para a execução de nossas celulás de código. As bibliotecas utilizadas foram:
#### Bibliotecas utilizadas
* **Tratamento da base de dados:**
 1. *Pandas*
 2. *Numpy*
 3. *json*
 4. *os*
 5. *Geopandas*
* **Visualização dos dados:**
 1. *Matplotlib*
 2. *Seaborn*
 3. *Folium*
* **Machine Learning:**
 1. *Scikit learn*
 2. *YellowBrick*
* **Exportação**
 1. *Files*
 2. *Imageio*

### MACHINE LEARNING

Temos em mão um problema de Classificação e como ja dito anteriormente a Classificação é uma técninca de aprendizado de máquina que objetiva prever rótulos de classe categóricas. O nosso problema é basicamente prever com base das caracteristicas dos pacientes, se ele deverá ir ou não para a UTI (ICU=1). A partir disso para esse projeto foram selecionados os seguintes modelos :

1. Random Forest Classifier;
2. Linear Regression;
3. Decision Tree Classifier;
4. Linear Discriminant Analysis.

Cada um desses modelos foram abastecidos com os dados e após realizado as suas predições tiveram uma avaliação baseada nas seguintes métricas:

* **Acurácia(ACC)**: A acurácia são os acertos de todo o meu universo de dados, ela irá determinar os acertos daquele universo, mas não de cada classe de dados. Ela por si só não nos permite uma visualização dos dados que o modelo acertou mais ou menos. A acurácia é uma das métricas mais utilizadas para mensurar o desempenho de modelos classificadores e sua formula pode ser expressa da seguinte forma:

* **Matriz de Confusão:** Uma forma de visualizar o universo de dados que foram acertados e os que não foram, é através da Matriz de Confusão. Essa matriz retorna visualmente os valores **Verdadeiros Positivos (VP)**, **Falsos Positivos (VP)**, **Falsos Positivos (VP)**,**Falsos Negativos (VP)**. Quanto mais próxima de 1 for os valores VN e VP e mais próxima de zero os valores de FN e FP, maior a acurácia do modelo. Os valores da Matriz de confusão nesse trabalho foram normalizados(escala de 0 a 1).E buscando facilitar o entendimento dos seus valores, vamos adapta-los para esse trabalho da seguinte forma:

<img src="Images/IMAGES/Matriz de Confusão.jpg"/>

* **Recall ou Sensibilidade**: Essa métrica busca entender a proporção dos valores que são de fato positivos e que foram preditos de forma correta. Sua formula pode ser descrita como a proporção entre os positivos verdadeiros (VP) e os falsos negativos (FN), conforme sua formula

* **Precision ou Precisão**: Essa métrica busca medir a proporção de predição positivas que estão corretas, ou seja, o quão bom um modelo prediz os valores positivos. É uma proporção entre os Verdadeiros Positivos e os Falsos Positivos. Busca entender *Quantos resultados relevantes são devolvidos?*. Podemos descrever matematicamente da seguinte forma:

* **F1-score**: Essa métrica pode ser entendida como a medida harmônica entre a precisão e o recall. O F1-score tenderá a ser alto quando ambos as métricas forem altas e similares, ou seja, o f1-score é uma espécie de alerta e demonstra quando alguma das medidas podem estar muito baixas. A sua representação matemática é a seguinte: 

* **CURVA ROC Receiver Operating Characteristic Curve**: Uma curva que demosntra o desempenho de algum modelo classificador binário, essa curva é obtida em razão da taxa de valores Verdadeiros Positivos (TPR) e pela taxa de Falsos Positivos (FPR)

* **AUC - Area Under The Curve**: O AUC como o nome diz, é uma área abaixo da curva ROC , seu valor varia entre 0 e 1, quanto mais próximo de 1, melhor é a performance do modelo. Abaixo irei demostrar casos de um AUC e de uma curva ROC:

<img src="Images/IMAGES/curva roc_auc2.png"/>

**Por fim, após todo o processo trago aqui um resumo das métricas de cada um dos modelos na tabela abaixo:**


| METRIC     |Random Forest|Logistic Regression|Random Tree|Linear Discriminat|
|:----------:|:-----------:|:-----------------:|:---------:|:----------------:|
|ACURÁCIA    | **76.06%**  |  73.24%           | 71.83%    | 70.42%           |
|PRECISION(0)| **0.72**    |  0.71             | 0.69      | 0.69             |
|PRECISION(1)| **0.83**    |  0.77             | 0.78      | 0.73             |
|RECALL(0)   | **0.89**    |  0.84             | 0.87      | 0.82             |
|RECALL(1)   | **0.61**    |  0.61             | 0.55      | 0.58             |
|F1.Score(0) | **0.80**    |  0.77             | 0.77      | 0.75             |
|F1.Score(1) | **0.70**    |  0.68             | 0.64      | 0.64             |
|AUC TEST    | **0.80**    |  0.76             | 0.68      | 0.76             |
|AUC TREINO  | **0.94**    |  0.83             | 0.88      | 0.86             |
|AUC         | **0.82**    |  0.78             | 0.75      | 0.74             |

        
<a name="es"></a>
## ESTRUTURA DO REPOSITÓRIO

Esse repositório, foi dividido por mim em 3 pastas, são elas:

### Data

Aqui você encontrará a base de dados usada nesse projeto, essa pasta se divide em:
* [**Data(Modified)**](https://github.com/Glaudemias/PROJETO_FINAL/tree/main/Data/Data(Modified)): Aqui estão as bases de dados ja tratadas, divididos em dados de Machine Learning e dados de Análise;
* [**Primary Data**](https://github.com/Glaudemias/PROJETO_FINAL/tree/main/Data/Primary%20Data): Aqui estão os dados brutos adiquiridos diretamente do [Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19);
* [**ShapeFiles**](https://github.com/Glaudemias/PROJETO_FINAL/tree/main/Data/ShapeFiles): Aqui estão os arquivos shape que me possibilitou gerar visualizações cartográficas.

### Images

Aqui você encontrará as Imagens e demais peças gráficas geradas ao longo desse projeto e usadas para exemplificar o arquivo read.me:
* [**GIFS**](https://github.com/Glaudemias/PROJETO_FINAL/tree/main/Images/GIFS): Aqui estarão os gifs gerados ao longo do projeto;
* [**HTML**](https://github.com/Glaudemias/PROJETO_FINAL/tree/main/Images/HTML): Arquivo HTML contendo o mapa interativo gerado a partir da biblioteca **folium**;
* [**IMAGES**](https://github.com/Glaudemias/PROJETO_FINAL/tree/main/Images/IMAGES): Aqui estarão as imagens geradas ao longo do projeto.

### Notebooks

Aqui você encontrará os notebooks que juntos formam todo o projeto, essa pasta se divide em:
* [**Notebooks**](https://github.com/Glaudemias/PROJETO_FINAL/tree/main/Notebooks): Os arquivos "ipynb" são referentes
* [**Parte_2_Análise_Exploratória_dos_Dados**](https://github.com/Glaudemias/PROJETO_FINAL/tree/main/Notebooks/Parte_2_An%C3%A1lise_Explorat%C3%B3ria_dos_Dados): Aqui temos o arquivo HTML do notebook, foi usada como uma terceira possibilidade de visualizar as plotagens interativas do projeto

<a name="concl"></a>
## CONCLUSÕES
Após a realização de cada projeto trarei aqui as conclusões de cada parte do projeto:
### Parte 1
Ao fim dessa parte, obtivemos duas bases de dados. Em uma tivemos a **redução do número de features altamente correlacionadas**, **sem dados faltantes** e prontas para serem utilizadas em algoritimos de Machine Learning. **Uma segunda é a base com os dados categóricos**, esses serão analisados e ajudaram a visualizar as informações da base de dados do Sírio Libanês.

### Parte 2
Ao fim dessa parte, obtivemos as seguintes conclusões da análise exploratória dos dados:
**Análise da base do COVID-19**
Segunda a análise eploratória podemos confirmar que:

- **São Paulo**, **Brasilia** são os municipios mais afetados no número de casos
- **Rio de Janeiro** se destaca no númeor de óbitos
- **A região Sudeste** é a região que mais possui municipios afetados pela pandemia do Corona Virus
- Provavelmente a proximidade de cidades com centros urbanos adensados e capitais 

**Analise da base do Sírio libanês**
Ao fim da análise exploratória podemos confirmar que a maior parte de pessoas que compõem esse banco de dados, são pessoas do gênero masculino abaixo de 60 anos e que não deram entrada na UTI. Contudo interessa pra gente saber o número de pessoas que entraram na UTI e são eles:

* **Genero: Masculino 61.93%**

* **Idade: Acima de 60 anos, principalmente acima de 90 anos**

* **Grupo de doenças: Excluindo a classe outras, as pessoas com hipertensão são as que mais deram entrada na UTI.**

*Esses números demonstram e comprovam o conhecimento que ja sabemos sobre a COVID-19, pessoas com doenças respiratórias e idosas são mais sucetíveis a complicações em caso de contaminação pelo  novo corona vírus.*

### Parte 3
Chegado ao fim dessa parte 3, concluimos assim esse projeto, com os seguintes resutlados:

* Foram testados 4 modelos de Classificação, a fim de encontrar o melhor desempenho na predição de pacientes que precisam ou não ser levados para UTI;

* Dentre todos os modelos testados obtivemos a seguinte classificação do melhor para o pior desempenho:

1. **Random Forest Classifier**

2. **Logistic Regression**

3. **Decision Tree Classifier**

4. **Linear Discriminant Analysis**

* O critério de desemepnho entre o primeiro e o segundo lugar foram os valores das métricas, onde o **Random Forest** se mostrou melhor que o **Logistic Regression**

* Os modelos foram explorados, investigados, abastecido com base nos dados tratados e na aplicação de seus hiprparâmetros

* Uma provavel razão para o desempenho dos dois primeiros modelos em relação ao 3º e 4º lugar, pode ser a quantidade de hiperparametros utilizados. Umas vez que esses ultimos tiveram menos opções que os primeiros.

**Random Forest**
 
Em relação a utilização do Random Forest Classifier para a realização dos objetivos propostos para esse projeto tivemos os seguintes resultados:

1. Capacidade do modelo em prever os pacientes  que serão adimitidos/necessitarão de um leito de UTI (Valores Verdadeiros Positivos da Matriz de Confusão):

O modelo obteve como desemepenho um acerto de **0.61** (valor normalizado) de pacientes que deveriam ir para UTI

2. Prever os pacientes que não serão adimitidos/necessitarão de um leito de UTI.

O modelo obteve como desemepenho um acerto de **0.89** (valor normalizado) de pacientes que não deveriam ir para UTI

**Acurrácia**: 76.06%
**AUC MÉDIO**: 0.82

| METRIC     |RANDOM FOREST    |
|:----------:|:---------------:|
|ACURÁCIA    | **76.06%**      | 
|PRECISION(0)| **0.72**        | 
|PRECISION(1)| **0.83**        | 
|RECALL(0)   | **0.89**        | 
|RECALL(1)   | **0.61**        | 
|F1.Score(0) | **0.80**        | 
|F1.Score(1) | **0.70**        |  
|AUC TEST    | **0.80**        |  
|AUC TREINO  | **0.94**        |  
|AUC         | **0.82**        |


### Para Avançar na Pesquisa

Pensando em possiveis estratégias que buscassém melhorar o modelo  e o próprio projeto penso que:

* É interessante explorar outros modelos de Classificação e comparar o resultado com o do Random Forest;
* Aumentar o número de Hiperparâmetros dos modelos de Decision tree Classifier e Linear Discriminant Analysis;
* Testar novas formas de separação de dados e diminuiçaõ de colunas da base de dados, perceber se a retirada das features melhora ou piora os modelos existentes;
* Investigar o porque de não adicionar o parâmetro `test_size` ao train test split, aumenta a performance e melhora bastante o resultado dos modelos utilizados.

Por fim é importante salientar que esse projeto deve ser o pontapé inicial para o aprofundamento no mundo de Machine Learning e buscar aperfeiçoar os conhecimentos em aprendizado de máquina. 

Muito Obrigado, por ter vindo até aqui!!😄😄

... Até logo 🖐️🖐️

<a name="ag"></a>
## AGRADECIMENTOS
Esse projeto como disse antes, foi proposto como resultado final do Bootcamp de Data Science aplicada, ofertado pela Alura. Tendo isso em mente gostaria de agradecer os professores e profissionais que nos acompanharam ao longo dessa trajetório de aprendizado: Guilherme, Thiago, Karol, Allan. Muito obrigado pelos ensinamentos transmitidos. Gostaria de agradecer também ao ScubaTeam por estarem sempre dispostos a ajudar e tirar dúvidas ao longo dos módulos e dos projetos.

Agradeço ainda aos meus amigos do Discord, Carol, Valquiria e Felipe, por toda a ajuda e pelas conversas que semrpe faziam meu projeto andar. Agradeço também meus amigos Erik e Stephane por sempre me incentivar e por me ajudar a deixar o readme mais bonitinho hahaha

OBRIGADO!! :)

<a name="rb"></a>
## REFERÊNCIAS BIBLIOGRÁFICAS
[1](https://covidreference.com/timeline_pt) - ***Linha do Tempo Covid-19 no mundo***

[2](https://www.who.int/docs/default-source/coronaviruse/situation-reports/20200304-sitrep-44-covid-19.pdf?sfvrsn=783b4c9d_2) - ***Coronavirus disease 2019 (COVID-19) Situation Report – 44***

[3](https://pt.wikipedia.org/wiki/Aprendizado_de_m%C3%A1quina) - ***Winkipedia - Aprendizado de Máquina***

[4](https://medium.com/data-science-brigade/a-diferen%C3%A7a-entre-intelig%C3%AAncia-artificial-machine-learning-e-deep-learning-930b5cc2aa42) - ***A diferença entre inteligência artificial Machine Learning e Deep Learning***

[5](https://paulovasconcellos.com.br/o-que-e-machine-learning-e-como-aprender-sem-gastar-nada-2e612f13102b) - ***O que é Machine Learning e como aprender sem gastar nada***

[6](https://medium.com/ensina-ai/por-que-usar-machine-learning-nos-seus-projetos-7aed3822962b) -  ***Por que usar machine learning nos seus projetos ?***

[7](https://medilab.net.br/2019/08/15/entenda-os-impactos-do-machine-learning-em-saude/) - ***Entenda os impactos do Machine Learning em saude***

[8](https://www.saudebusiness.com/voc-informa/hospital-aumenta-eficincia-com-gesto-orientada-dados) - ***Hospital aumenta a eficiência com gestão orientada a dados***

[9](https://didatica.tech/underfitting-e-overfitting/) - **Underfitting e Overfitting**

[10](https://pt.wikipedia.org/wiki/Hospital_S%C3%ADrio-Liban%C3%AAs#cite_note-2) - **Sírio Libanês Wikipedia**

[11](https://gauchazh.clicrbs.com.br/colunistas/daniel-scola/noticia/2017/12/diretor-do-sirio-libanes-futuro-dos-hospitais-sera-igual-ao-das-farmacias-so-ficarao-as-grandes-redes-cjaxwodqq098n01mkouwvaskg.html) - **Diretor do Sírio-Libanês: "Futuro dos hospitais será igual ao das farmácias, só ficarão as grandes redes"**

[12](https://www.hospitalsiriolibanes.org.br/iep/mestrado-doutorado/Paginas/default.aspx) - **Mestrado e Doutorado**

[13](https://eadsiriolibanes.org.br//) - **EAD Sírio Libanês**

[14](https://saude.estadao.com.br/noticias/geral,primeiro-estudo-com-hidroxicloroquina-no-pais-tera-resultado-em-dois-meses,70003248172) - **Primeiro estudo com hidroxiclorquina no pais terá resultado em dois meses**

[15](https://especiais.g1.globo.com/bemestar/vacina/2021/mapa-brasil-vacina-covid/) - **Mapa de vacinação da COVID-19 no Brasil**

## CONTATO
Se você tiver alguma sugestão ou dica para melhorar o projeto, pode me mandar uma mensagem aqui:

[![Linkedin Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white&link=https://www.linkedin.com/in/glaudemias-grangeiro-junior-940a951a1/)](https://www.linkedin.com/in/glaudemias-grangeiro-junior-940a951a1/)

[<img src="https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=Gmail&logoColor=white" />](mailto:glaudemias.arqurb@gmail.com)

Por fim e antes que me esqueça...
  <p align="rigth">
  <img width="360" height="460" alt="Zé gotinha metendo o brega funk cabuloso" src="/Images/GIFS/ezgif.com-gif-maker.gif" />
     ...viva o SUS :)
