# PROJETO_FINAL

# **INTRODUÇÃO**

Se apresentar

## COVID - 19
	Desde quando foi noticiado o primeiro caso de pessoa infectada pelo vírus coronavírus (SARS-COV 2), grandes têm sido os esforços da comunidade científica e  de alguns líderes mundiais, na contenção e achatamento da propagação da doença. Por ser um vírus de disseminação comunitária, medidas como o isolamento social e distanciamento social foram levantadas no dia 25 de Fevereiro de 2020 pelo relatório de pesquisadores chineses, mostrando que as medidas de quarentena adotadas pelo país havia sido  o primeiro a ser contaminado pela pandemia e o primeiro a tomar medidas de restrição[1] 
  
	No Brasil o  primeiro caso de covid-19 foi registrado em 06/02/2020, na cidade de São Paulo. Em 30 de janeiro de 2020, a Organização Mundial da Saúde (OMS) declarou a epidemia emergência internacional, o mundo teve que se reorganizar e tentar conviver com uma nova realidade marcada por incertezas, mortes, depositando esperança na criação da vacina. Hoje com a vacinação em andamento desde 08 de dezembro de 2020 (primeira vacinada no mundo), já se vislumbra uma possível saída de todo esse caos.[2]  
  
Contudo a pandemia não está nada perto de passar, as medidas de contenção como o isolamento social são importantes para o achatamento da curva de contaminação e consequentemente evitar a crise hospitalar, causada pela demanda superior a capacidade do sistema de saúde. Sendo assim, preparar nosso sistema de saúde para melhor prover um atendimento e demanda por leitos e atendimentos, é essencial. E nessa perspectiva dados individualizados baseados em atendimentos já feitos podem auxiliar na demanda de leitos de Unidade Intensiva de Tratamento, EPI e profissionais disponíveis para um atendimento seguro e eficaz.      

## MACHINE LEARNING

Machine Learning é uma área da ciência e engenharia da computação, que trata da capacidade automatizada de se ensinar uma máquina, esse aprendizado pode ser supervisionado ou não[3]. Através da elaboração de algoritmos é possível coletar dados, aprender com eles e então fazer uma predição, determinação sobre alguma coisa no mundo [4]. As aplicações de Machine Learning são as mais variadas, desde ler e identificar um e-mail como spam ou não, até visão computacional e como dito anteriormente o tipo de aprendizagem dessa máquina se divide em dois tipos principais, aqui serão definidos com base no escrito de [5] e [6]:

* **Aprendizado não  supervisionado**:   
	Quando falamos de um aprendizado de Máquina não supervisionado, não temos um resultado esperado como na elaboração de um modelo que deseja saber a melhor rota de um carro, nesse caso não existe um resultado esperado ou resposta certa. O aprendizado não supervisionado nos permite obter um agrupamento de conjuntos, mesmo sem estarem categorizados, por outro lado, por ser subjetivo, torna-se difícil definir métricas de quão boa está a performance do modelo.
	A técnica de agrupamento é chamada **clustering** sua vantagem se dá pelo fato de a máquina não ficar limitada aos rótulos dados pelo dataset, então é capaz de formar grupos por si só, levando em consideração as características mais importantes
* **Aprendizado supervisionado**:
	Em problemas de aprendizagem supervisionada é necessário que haja um conjunto de dados rotulados, que representam o conjunto de dados de características, por exemplo se o conjunto de dados é sobre características de um animal (tamanho dos olhos, cor do pelo, peso e tamanho), é necessário também rotular o conjunto dessas características, para que o algoritmo a ser utilizado faça a relações entre esses dados, sendo capaz de classificar gatinhos que não estejam rotulados. Os problemas de classificação são divididos em dois tipos: **Regressão** e **Classificação**, para explicar cada um, é necessário saber o conceito de tipos de rótulos, podem ser: **Numérica Contínuo**(datas, tamanhos e comprimentos) ou **Categórica**(gênero, tipo de material e método de pagamento).
	**Classificação** A classificação utiliza o rótulo do tipo dados  categóricos, já que seu objetivo é determinar a classe a que o conjunto pertence. A **Regressão** utiliza o rótulo do tipo numérico, que o permite estimar o valor baseado nas características do elemento, logo a saída desse modelo é também do tipo numérico contínuo. 
  
## MACHINE LEARNING NA SAÚDE
Entendendo que o potencial do aprendizado de máquina está diretamente ligado com a adição de dados a um modelo, não demorou para que se utilizassem os benefícios providos por essa tecnologia  nas diversas áreas da sociedade, dentre elas a saúde.Por meio de equipamentos e algoritmos que se baseiam em Machine Learning, é possível cruzar os dados clínicos dos pacientes com as referências epidemiológicas obtidas nos sites governamentais em saúde e formular diversas hipóteses sobre um evento clínico. Dessa forma o uso de Machine Learning na saúde fornecerá conclusões mais apuradas do que se fossem analisadas isoladamente, trazendo mais segurança ao profissional de saúde quando tomar as decisões pelo paciente. []7
De acordo com [8] as aplicações possíveis para a utilização de Machine Learning na área da saúde são:

* **Direcionamento:** Uma das preocupações decorrentes do novo coronavírus é o isolamento de pessoas sintomáticas para reduzir o contágio e propagação da doença. Existem processos inteligentes capazes de relacionar sintomas apresentados com a doença, resultando em direcionamento mais ágil dos infectados;
* **Diagnostico de Câncer:** Ao analisar as características de um nódulo na imagem de ressonância, segundo um banco de dados já definido, é possível classificar o nódulo como maligno ou benigno. Também é possível saber a probabilidade de o nódulo pertencer a cada uma das classificações, para então definir o diagnóstico do paciente e os próximos passos do tratamento;

* **Triagem:** O uso de machine learning na saúde também pode não ser tão específico, indicando não apenas uma, mas as possíveis enfermidades que o paciente pode ter de acordo com os sintomas que ele apresenta. Uma aplicação possível seria durante a triagem: ao chegar no pronto atendimento, diante dos sintomas apresentados e com o auxílio dos dados já pré-definidos no sistema, o paciente é encaminhado ao especialista mais adequado;
**Entendimento do contexto:** aplicativos de pesquisa podem contribuir para a compreensão do avanço de pandemias, como a da COVID-19, por meio de testes não invasivos. Por exemplo, as pessoas respondem sobre seu quadro clínico atual e, com o passar do tempo, é possível avaliar o avanço da doença na região. Neste sentido, é possível agir preventivamente quanto à disponibilidade de equipamentos e profissionais para atender às diferentes demandas regionais
Do ponto de vista político administrativo a ciência de dados e o Machine Learning, estão se mostrando  ferramentas essenciais para lidar com a pandemia de Coronavírus, já que as ferramentas permitem um maior monitoramento, leitura da realidade e possibilidade de predição dos números de casos. Mas do ponto de vista do tratamento de pacientes acometido pela síndrome respiratória causada pelo novo Coronavírus as técnicas de Machine Learning, podem auxiliar na predição de leitos de hospitais, atendimento por equipe médica, EPI e tudo o que for necessário para o tratamento da doença, a partir das experiências passadas e das base de dados construída por hospitais ao longo do tempo [9].

# **DADOS**
	Os dados desse notebook foram retirados da plataforma [Kaggle](), são referentes aos dados de internação do hospital Sírio Libanês.
Explicar de onde foram tirados os dados
Quais são as suas colunas
Descrever a estrutura do banco de dados
Citar quais colunas que devem sair por conta do Machine Learning (criar o link para o tópico de machine learning)
os dataframes que foram utilizados

# **METODOLOGIA**
(breve descrição)

## Perguntas norteadoras, ou hipotéses que o Projeto buscou responder(decidir nome)
Trazer as perguntas para ca

##  **FERRAMENTAS**

Quais bibliotecas foram utilizadas
Para que as bibliotecas foram utilizadas
Qual a versão do python e qual IDE utilizada

## **MACHINE LEARNING

Explicar o Cross Validation
Explicar os modelos analisados
Explicar as métricas utilizadas
Explicar quais dados sairam 
Explicar Overfitting

#ESTRUTURA DO REPOSITÓRIO
Esse repositório, foi dividido por mim em 3 pastas, são elas:

## **Data** 
Explicar o que é e o que contém

## **Images**
Explicar o que é e o que contém

## **Notebooks**
Explicar o que é e o que contém 

# **CONCLUSÕES**
Quais as conclusões foram obtidas, imagens e tópicos
Os resultados 
Avançar na pesquisa

# **AGRADECIMENTOS**

# ** REFERÊNCIAS**
[1](https://covidreference.com/timeline_pt) - Linha do Tempo Covid-19 no mundo
[2](https://www.who.int/docs/default-source/coronaviruse/situation-reports/20200304-sitrep-44-covid-19.pdf?sfvrsn=783b4c9d_2) - Coronavirus disease 2019 (COVID-19) Situation Report – 44 
[3](https://pt.wikipedia.org/wiki/Aprendizado_de_m%C3%A1quina) - Winkipedia
[4](https://medium.com/data-science-brigade/a-diferen%C3%A7a-entre-intelig%C3%AAncia-artificial-machine-learning-e-deep-learning-930b5cc2aa42)- Medium 
[5](https://paulovasconcellos.com.br/o-que-e-machine-learning-e-como-aprender-sem-gastar-nada-2e612f13102b) - Paulo Vasconcellos
[6](https://medium.com/ensina-ai/por-que-usar-machine-learning-nos-seus-projetos-7aed3822962b) -  Por que usar machine learning nos seus projetos ?
[7](https://medilab.net.br/2019/08/15/entenda-os-impactos-do-machine-learning-em-saude/) 
[8] - (https://www.saudebusiness.com/voc-informa/hospital-aumenta-eficincia-com-gesto-orientada-dados)

