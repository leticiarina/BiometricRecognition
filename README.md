
  
# Reconhecimento Biométrico Baseado em Imagens das Mãos
Projeto final da disciplina SCC0251 - Processamento de Imagens, ministrada por Moacir Ponti no primeiro semestre de 2018 para o curso de Ciências da Computação - ICMC/USP.

## Autores
[Letícia Rina Sakurai](https://github.com/leticiarina/) (9278010)

[Rafael Augusto Monteiro](https://github.com/RafaelMonteiro95/) (9293095)

## Projeto
### Tema
Biometria. Reconhecimento biométrico com base em imagens das mãos de um indivíduo.

### Resumo
O objetivo do trabalho é implementar um método que permita a identificação biométrica com base em imagens da mão de um indivíduo. O sistema recebe como entrada uma foto da mão de um indivíduo, e verifica se ele está cadastrado no sistema com base na similaridade entre 20 características extraídas de cada imagem do banco de dados. 

## Introdução
A biometria é um mecanismo de identificação mais seguro, ao utilizar características físicas singulares de cada indivíduo, como olhos, impressões digitais e palma das mãos [1]. Porém, sistemas biométricos, em sua maioria, utilizam dispositivos específicos para a captura de informações, como leitores de impressões digitais e scanners de palma de mão. A necessidade de tais sensores pode tornar mais cara ou complexa a implementação de um sistema biométrico.

A utilização de dispositivos mais acessíveis, como smartphones, torna possível que usuários que não tem acesso à sensores altamente especializados desenvolvam sistemas de segurança biométricos modestos. Ainda, a utilização de imagens das mãos é menos intrusiva que outros métodos, como capturas de íris, causando menos desconforto aos usuários. Portanto, neste projeto, foi desenvolvido um sistema de identificação biométrica com base em 20 características extraídas de fotos das mãos, capturadas de smartphones.

## Conjunto de Dados
A base de dados utilizada consiste em 5 imagens da mão esquerda de 15 indivíduos. As fotos foram capturadas com o smartphone Xiaomi Redmi Note 5A, que possui uma câmera de 13 Megapixels. Para evitar ruídos, as fotos foram feitas com um tecido preto no plano de fundo. Ainda, para manter um padrão de distância entre o usuário e a câmera, foi utilizado um apoio de 15cm. A imagem abaixo ilustra o ambiente de captura:

inserir imagem

As imagens estão disponíveis na pasta [handDatabase](handDatabase/). Cada imagem está no formato:
```
<usuário>_<nro_imagem>.jpg
```
onde <usuário> é o nome de um usuário e <nro_imagem> é um número de 1 a 5. O dataset foi dividido em imagens para o banco de dados de usuários (imagens de número 1 a 4) e em imagens para validação (imagens de número 5). Mais detalhes sobre as imagens estão disponíveis nas sessões seguintes.
## Métodos

### Visão Geral 
(Uma demonstração dos métodos descritos pode ser vista no notebook [biometricRecognition.ipynb](biometricRecognition.ipynb).)

Para realizar o acesso biométrico, o sistema deverá verificar se a imagem fornecida da mão do usuário possui similaridade acima de um determinado threshold com algum dos registros guardados no banco de dados. 

A imagem é convertida para uma forma binária, preto e branca. Em seguida, é feita a segmentação da mão, isolando os componentes mais relevantes. São extraídos e utilizados na comparação 20 atributos:


Nro. A. | Nome do Atributo | Nro. A.| Nome do Atributo
--- | --- | --- | ---  
1| largura da palma | 11 | largura meio dedo 1
2| comprimento da palma | 12 | largura ponta dedo 1
3| largura da mão | 13 | comprimento dedo 2
4| comprimento da palma | 14 | largura base dedo 2
5| comprimento dedo 0 | 15 | largura meio dedo 2
6| largura base dedo 0 | 16 | largura ponta dedo 2
7| largura meio dedo 0 | 17 | comprimento dedo 3
8| largura ponta dedo 0 | 18 | largura base dedo 3
9| comprimento dedo 1 | 19 | largura meio dedo 3
10| largura base dedo 1 | 20 | largura ponta dedo 3


Os dedos utilizados são indicador, médio, anelar e mindinho, descartando o dedão. 
Por fim, é feita a normalização dos valores extraídos pelo comprimento da mão, e as comparações são feitas entre a mão do usuário e as mãos cadastradas no banco de dados. Caso a similaridade entre a mão do usuário e a mão cadastrada no banco de dados seja maior que um limiar previamente calculado para aquele usuário, o usuário é reconhecido.

As sessões seguintes explicarão em mais detalhes as etapas do sistema.

### Preprocessamento e Segmentação
É importante ressaltar que boa parte das operações utilizadas nesta sessão são implementações da biblioteca OpenCV para Python. Portanto, a implementação de algoritmos específicos não será detalhada.

As imagens utilizadas são capturadas em cores. Para realizar a segmentação dos objetos da imagem, é interessante trabalhar com máscaras binárias preto-e-brancas. Portanto, as imagens são convertidas para escala de cinza. Em seguida, é encontrando um limiar pelo algoritmo de Otsu. O limiar é aplicado, gerando uma máscara binária.

exemplo1

Em seguida, para remover ruídos do fundo, é feita a busca por todos os contornos encontrados na imagem. O contorno de maior área é mantido, enquanto os demais são descartados. Assim, apernas o contorno da mão, que deve ser o destaque da imagem, permanece na máscara resultante

exemplo 2

Em seguida, é gerado o *convex hull* do objeto na máscara. O *convex hull* encobre o objeto na máscara com um polígono convexo. É possível, então, encontrar todos os defeitos entre o *convex hull* e o objeto na máscara. São mantidos apenas os defeitos que possuem uma distância entre o objeto da máscara e o *convex hull* acima de 4000, limiar definido empiricamente com base em testes realizados nas imagens do dataset.

exemplo3

Para realizar a segmentação entre a palma e as demais partes da mão, é feito um recorte circular na palma da mão. O círculo utilizado é o menor círculo que encobre todos os defeitos encontrados na etapa anterior. Após o corte, uma máscara com a palma e outra com as demais partes da mão são geradas.

exemplo4

Em seguida, são explorados os contornos dos objetos restantes na máscara com as demais partes da mão. Em um primeiro passo, é excluído o braço, sendo considerado o contorno com a menor altura. É gerada uma máscara com os cinco dedos. Em seguida, é excluído o dedão, sendo o objeto com a maior área, considerando o retângulo que o encobre. É gerada uma máscara com os quatro dedos

exemplo5

No próximo passo, os contornos restantes da máscara de quatro dedos são analisados. Os quatro dedos são recortados e rotacionados, gerando quatro máscaras binárias onde cada dedo aponta para cima.

exemplo6

### Extração de Atributos

Todos os atributos são extraídos com base em retângulos que encobrem os objetos das máscaras geradas anteriormente. Os retângulos são obtidos diretamente por funções do OpenCV.

Para extrair os atributos da mão, é feita a união (soma) das máscaras de cinco dedos e palmas. Em seguida, é gerado o contorno do novo objeto, e é feita a medição da altura e da largura da mão com base no menor retângulo que encobre a mão (altura e largura). O mesmo foi feito para extrair a altura e largura da palma.

exemplo7
exemplo8

Cada dedo gera quatro atributos: um de comprimento, e três de largura. Para extrair o comprimento, foi utilizada a altura da caixa que encobre o dedo. As três larguras (base, meio e ponta do dedo) foram calculadas nos pontos de altura 25%, 50% e 75% da caixa. Portanto, foram contados o número de pixeis brancos nas linhas da máscara posicionadas nas alturas 0.25*h, 0.5*h e 0.75*h, onde h é a altura da caixa que encobre a mascara do dedo.

exemplo9

### Comparação
Os atributos de todas as imagens foram carregados em um *dataframe* do Pandas e armazenados em um arquivo csv. Para minimizar os efeitos de diferença de ângulos nas fotos, todos os valores foram normalizados pelo comprimento da mão. Assim, todos os atributos (com excessão do comprimento da mão) possuem valores sempre menores que 1.

Como métrica de distância entre os documentos, utilizamos a distância euclidiana. 

## Resultados
* Expandir dataset para pelo menos 5 imagens de 15 indivíduos.
* Implementação dos métodos de extração de atributos
* Implementação dos métodos de comparação

## Demonstração
- Um exemplo dos métodos implementados está disponível no arquivo [biometricRecognition.ipynb](biometricRecognition.ipynb).
- Todas as funções definidas no arquivo [biometricRecognition.ipynb](biometricRecognition.ipynb) também estão implementadas no arquivo [biometricRecognition.py](biometricRecognition.py).
