# Reconhecimento Biométrico Baseado em Imagens das Mãos
Projeto final da disciplina SCC0251 - Processamento de Imagens, ministrada por Moacir Ponti no primeiro semestre de 2018 para o curso de Ciências da Computação - ICMC/USP.

## Autores
Letícia Rina Sakurai (9278010)

Rafael Augusto Monteiro (9293095)

## Projeto
### Tema
Biometria. Reconhecimento biométrico com base em imagens das mãos de um indivíduo.

### Descrição
O objetivo do trabalho é implementar um método que permita a identificação biométrica com base em imagens da mão de um indivíduo. Para isso, o sistema receberá imagens da mão de uma pessoa, aplicar as transformações necessárias, extrair características úteis para o reconhecimento e escolher o registro cadastrado com características mais similares à amostra fornecida.

## Imagens utilizadas
A base de dados usada foi gerada pelos próprios autores e está disponível na pasta [handDatabase](handDatabase).

## Métodos
* Transformação da imagem de RGB para escala de cinza e thresholding pelo algoritmo de Otsu

### A implementar
* Extração de características das imagens (bordas, medida dos dedos, espaçamento entre os dedos);
* Reconhecimento da pessoa por meio do cálculo da diferença entre as imagens.