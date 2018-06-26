
# Reconhecimento Biométrico Baseado em Imagens das Mãos
Projeto final da disciplina SCC0251 - Processamento de Imagens, ministrada por Moacir Ponti no primeiro semestre de 2018 para o curso de Ciências da Computação - ICMC/USP.

## Autores
[Letícia Rina Sakurai](https://github.com/leticiarina/) (9278010)

[Rafael Augusto Monteiro](https://github.com/RafaelMonteiro95/) (9293095)

## Projeto
### Tema
Biometria. Reconhecimento biométrico com base em imagens das mãos de um indivíduo.

### Descrição
O objetivo do trabalho é implementar um método que permita a identificação biométrica com base em imagens da mão de um indivíduo. Para isso, o sistema receberá imagens da mão de uma pessoa, aplicar as transformações necessárias, extrair características úteis para o reconhecimento e escolher o registro cadastrado com características mais similares à amostra fornecida.

## Imagens utilizadas
A base de dados usada foi gerada pelos próprios autores e está disponível na pasta [handDatabase](handDatabase). 
Atualmente, a base de dados conta com 2 imagens da mão esquerda de dois indivíduos. As imagens foram capturadas utilizando as câmeras dos smartphones *Redmi Note 5A* e *Iphone 7*.

## Métodos

### Visão Geral 
O sistema deve receber uma imagem da mão de um usuário, capturada com um smartphone. Em seguida, a imagem é convertida para escala de cinza e transformada em uma imagem binária (preto e branco) para diferenciar a mão do fundo da imagem.

Obtém-se o contorno da mão, os pontos principais - vãos entre os dedos, detecção da transição entre palma e punho -, que são utilizados para definir a palma da mão e separar os dedos em diferentes máscaras. 

O próximo passo é realizar a medição do comprimento e da largura dos dedos. A largura dos dedos é medida em três pontos diferentes: na ponta, no meio e na base. O dedo polegar não é utilizado.

Feita a medição dos dedos, é feita a análise da palma, extraindo sua largura, comprimento e proporção. Ainda, é extraído o comprimento da mão.

Por fim, todos os 20 atributos obtidos (4 de cada dedo, 4 da mão) são unidos em um vetor. Esse vetor é comparado aos vetores previamente extraídos e armazenados. Caso o vetor obtido possua similaridade acima de um determinado threshold com outro vetor, o usuário atual é reconhecido.

### Métodos Implementados
#### Pré Processamento
* Transformação da imagem de RGB para escala de cinza
* Thresholding e conversão para imagem binária pelo algoritmo de Otsu
* Eliminação de braço e outros ruídos no background
* Recorte de regiões sem conteúdo
* Segmentação da mão (extração dos dedos e da palma)

### Métodos Não Implementados
#### Extração de Atributos
* Extração de atributos dos dedos
* Extração de atributos das palmas
#### Comparação
* Definição do threshold de similaridade para matching
* Reconhecimento por meio do cálculo da diferença entre as imagens.

## Próximas Etapas
* Expandir dataset para pelo menos 5 imagens de 15 indivíduos.
* Implementação dos métodos de extração de atributos
* Implementação dos métodos de comparação

## Exemplos
- Um exemplo dos métodos implementados está disponível no arquivo [biometricRecognition.ipynb](biometricRecognition.ipynb). 
