# **Universidade Federal de Sergipe (UFS)**

## **Programa de Pós-Graduação em Ciência da Computação (PROCC)**

## **Disciplina: Tópicos Avançados em Engenharia de Software e Sistemas de Informação**

## **Data: 09/04/2026**

## **Aluno: Paulo A Costa  <-- Membro da Equipe 4**







# **Avaliação comparativa das respostas de 3 modelos para os quais foram submetidas 10 questões abertas da prova da 2ª fase do 42º Exame da OAB**



## DATASET QUESTÕES ABERTAS

* Formado por 5 questões sobre Direito Constitucional e 5 questões sobre Direito do Trabalho da prova do 42º Exame da OAB.



# 1\) Orientações para uso no Google Colab

## Arquivos de entrada esperados

No ambiente de terminal do Colab, coloque estes arquivos na pasta "/content":

* Arquivo com as 10 questões designadas para Paulo, membro da equipe 04

  * "equipe4\_paulo\_questoes\_abertas\_linhas\_131\_140\_template\_role\_content.jsonl"
* Arquivo das respostas de referência do gabarito espelho da prova do 42º Exame da OAB

  * guidelines.jsonl



## Arquivo principal

Uma vez o ambiente e os modelos preparados conforme arquivo "Paulo\_questoes\_abertas.ipynb". Executar os seguintes programas:

* program\_01.py
* program\_02.py



## Ajustes importantes

* Edite nos programas:

  * "OLLAMA\_BASE\_URL": URL acessível pelo Colab para API do Ollama
  * "OLLAMA\_MODELS": lista dos modelos para os quais o program\_01.py irá submeter as questões e calcular o BERTScore
  * "BERTSCORE\_MODELS": avaliadores BERTScore usados na comparação das respostas dos modelos com as respostas de referência do gabarito espelho da prova do 42º Exame da OAB



## Como ler a métrica

* A métrica principal do BERTScore é **F1**

  * "F1" em escala de 0 a 1
  * "F1 (%)" = "F1 \* 100"
* Exemplo:

  * "0.875" = "87.5%"



## Arquivos de saída

* "respostas\_modelos\_equipe4\_paulo.jsonl": respostas geradas pelos modelos
* "respostas\_modelos\_equipe4\_paulo.csv": respostas em CSV
* "avaliacao\_bertscore\_long\_equipe4\_paulo.csv": avaliação detalhada em formato longo (**principal**)
* "avaliacao\_bertscore\_wide\_equipe4\_paulo.csv": avaliação pivoteada para facilitar comparação
* "resumo\_modelos\_bertscore\_equipe4\_paulo.csv": resumo consolidado
* "README\_all\_results.md": relatório em markdown
* gráficos PNG



## Observações

* Para ser aprovado na 2ª fase (prova prático-profissional) do 42º Exame da OAB, a média mínima necessária é de 6,0 (seis) pontos no total.
* A prova prático-profissional vale um total de 10 pontos e é composta por:

  * Peça Processual: Vale 5,0 pontos.
  * Questões Discursivas: Quatro questões, valendo 1,25 ponto cada, totalizando 5,0 pontos.
* Nesse experimento usamos 10 questões (duas de peça processual e oito do tipo discursiva).
* Pontos importantes sobre as questões abertas:

  * A questão do tipo discursiva geralmente é dividida em dois itens (A e B).
  * A pontuação de cada questão é fracionada, avaliando o conhecimento técnico e a fundamentação legal.
  * É necessário, no mínimo, obter a nota total de 6,0 pontos somando a nota da peça processual mais as quatro questões discursivas.



# 2\) Fundamentação Teórica

* BERT
É o algoritmo Bidirectional Encoder Representations from Transformers (BERT), que define a arquitetura lógica da sua rede neural.  O seu trabalho é transformar palavras em embeddings (vetores numéricos) que capturam o contexto, usando os mecanismos de Transformer Encoder com Self-Attention. A similaridade do cosseno será calculada sobre esses vetores resultados do BERT.
* BERTScore
É um um método matemático (algoritmo de cálculo) de métrica de avaliação utilizado para medir/pontuar a qualidade de textos gerados por IA em uma escala de 0 a 1. O BERTScore utiliza a "inteligência" de modelos como o mBERT ou BERTimbau para comparar semanticamente o significado entre dois textos. O BERTScore é uma ferramenta de avaliação automatizada que se utiliza de modelos avaliadores de IA para calcular o F1-Score, através da "similaridade do cosseno" entre os vetores (embeddings) de cada token (palavra ou sub-palavra) da resposta gerada, contra os tokens da resposta de referência. A similaridade de cosseno mede o "ângulo" entre os vetores: quanto menor o ângulo, mais parecidos são os significados.
* F1-Score
O F1-Score (ou medida F1) é uma métrica estatística usada para medir a precisão de um modelo, sendo amplamente utilizada no meio jurídico porque ser "rigorosa". No Direito, onde uma inversão da polaridade de sentido semântico pode mudar o destino de um processo, o F1-Score é importante por equilibrar dois outros conceitos vitais que são Precisão (Precision) e Revocação/Sensibilidade (Recall), a fim de evitar e esquecer falsos positivos e negativos. \*\*Em nosso experimento calculamos o F1-Score das respostas dos 3 modelos testados, usando tanto mBERT como BERTimbau, tendo sido o avaliador BERTimbau que atribuiu maiores valores.
* mBERT
O mBERT ou Multilingual BERT é uma instância específica da arquitetura BERT que passou pelo processo de treinamento, usando simultaneamente um conjunto massivo de 104 idiomas, incluindo o português. Analogamente seria como um professor poliglota.
* BERTimbau
O BERTimbau Base é um modelo BERT pré-treinado para português brasileiro que alcança desempenho de última geração em três tarefas de PNL (Processamento de Linguagem Natural): Reconhecimento de Entidades Nomeadas, Similaridade Textual de Sentenças e Reconhecimento de Implicação Textual. Está disponível em dois tamanhos: Base e Grande. No domínio jurídico brasileiro, o BERTimbau costuma superar o mBERT, pois ele foi treinado exclusivamente com textos em português, captando melhor as nuances da nossa gramática e termos legais do que o modelo multilingual treinado em mais 100 línguas ao mesmo tempo (https://github.com/neuralmind-ai/portuguese-bert) (https://huggingface.co/neuralmind/bert-base-portuguese-cased). Analogamente seria como um professor específico de português.



# 3\) Resultados

# Avaliação comparativa de 3 modelos que responderam 10 questões da prova da 2ª fase do 42º Exame da Ordem dos Advogados do Brasil (OAB)



## Modelos testados

* llama3.1:8b
Llama é a família do primeiro modelo de código aberto que rivaliza com os melhores modelos de IA em termos de recursos de ponta em conhecimento geral, capacidade de direção, matemática, uso de ferramentas e tradução multilíngue. A versão atualizada do modelo 8B é multilíngue e possue um comprimento de contexto significativamente maior, de 128K, além de utilizarem ferramentas de última geração e apresentarem capacidades de raciocínio mais robustas. Isso permite que os modelos mais recentes da Meta suportem casos de uso avançados, como sumarização de textos longos, agentes conversacionais multilíngues e assistentes de codificação.
* gemma3:12b
Gemma é uma família de modelos leves do Google, construída com a tecnologia Gemini. Os modelos Gemma 3 são multimodais — processando texto e imagens — e apresentam uma janela de contexto de 128K com suporte para mais de 140 idiomas. Disponíveis com tamanhos de parâmetros de 270M, 1B, 4B, 12B e 27B, eles se destacam em tarefas como resposta a perguntas, sumarização e raciocínio, enquanto seu design compacto permite a implementação em dispositivos com recursos limitados.
* jurema:7b
Jurema 7B é o primeiro LLM jurídico brasileiro de código aberto, treinado com dados nacionais e criado por Escavador e NeuralMind, com apoio da FINEP, para atuar no Direito brasileiro (https://blog.escavador.com/jurema-7b-o-primeiro-llm-juridico-brasileiro-de-codigo-aberto-treinado-com-dados-nacionais#/).
Portanto, Jurema-7B é um LLM especializado no domínio jurídico brasileiro, criado a partir do ajuste fino do modelo Qwen2.5-7B-Instruct. O ajuste fino foi realizado com a utilização de um conjunto de dados sintético, majoritariamente com exemplos no formato de perguntas e respostas (Q\&A), embora também incluindo outros estilos de tarefas. Os exemplos foram compilados de uma coleção diversificada e curada de documentos jurídicos de alta qualidade, selecionados por sua representatividade, qualidade e diversidade (https://huggingface.co/Jurema-br/Jurema-7B). O repositório oficial do Hugging Face acusa, sem muitos detalhes, que **no Exame da OAB de 2023 ele obteve o score de 0,6840**. Não especifica, porém, até que ano correspondia o conjunto de dados da OAB no qual ele fora treinado.



## Ranking final dos modelos testados (avaliador BERTimbal)

|**Modelo**|**Quantidade Questões**|**F1-Score (%)**|
|-|-:|-:|
|llama3.1:8b|10|64.5|
|gemma3:12b|10|63.6|
|jurema:7b|10|61.8|



## Resumo por modelo e avaliador

|**Modelo**|**Modelo BERTScore**|**Quantidade Questões**|**F1-Score (%)**|
|-|-|-:|-:|
|llama3.1:8b|BERTimbau\_Large|10|64.5|
|gemma3:12b|BERTimbau\_Large|10|63.6|
|jurema:7b|BERTimbau\_Large|10|61.8|
|llama3.1:8b|mBERT|10|62.5|
|gemma3:12b|mBERT|10|59.4|
|jurema:7b|mBERT|10|57.6|



## Desempenho por área de especialidade

|**Modelo**|**Modelo BERTScore**|**Área Especialidade**|**Quantidade Questões**|**F1-Score (%)**|
|-|-|-|-:|-:|
|gemma3:12b|BERTimbau\_Large|Direito do Trabalho|5|64.1|
|gemma3:12b|BERTimbau\_Large|Direito Constitucional|5|63.2|
|gemma3:12b|mBERT|Direito do Trabalho|5|59.5|
|gemma3:12b|mBERT|Direito Constitucional|5|59.3|
|jurema:7b|BERTimbau\_Large|Direito do Trabalho|5|62.9|
|jurema:7b|BERTimbau\_Large|Direito Constitucional|5|60.8|
|jurema:7b|mBERT|Direito do Trabalho|5|57.7|
|jurema:7b|mBERT|Direito Constitucional|5|57.6|
|llama3.1:8b|BERTimbau\_Large|Direito Constitucional|5|64.5|
|llama3.1:8b|BERTimbau\_Large|Direito do Trabalho|5|64.5|
|llama3.1:8b|mBERT|Direito Constitucional|5|63.6|
|llama3.1:8b|mBERT|Direito do Trabalho|5|61.4|



## Desempenho por nível de dificuldade

|**Modelo**|**Modelo BERTScore**|**Nível Dificuldade**|**Quantidade Questões**|**F1-Score (%)**|
|-|-|-|-:|-:|
|gemma3:12b|BERTimbau\_Large|Médio|6|64.4|
|gemma3:12b|BERTimbau\_Large|Difícil|4|62.4|
|gemma3:12b|mBERT|Difícil|4|61.1|
|gemma3:12b|mBERT|Médio|6|58.2|
|jurema:7b|BERTimbau\_Large|Médio|6|62|
|jurema:7b|BERTimbau\_Large|Difícil|4|61.5|
|jurema:7b|mBERT|Difícil|4|59.9|
|jurema:7b|mBERT|Médio|6|56.1|
|llama3.1:8b|BERTimbau\_Large|Difícil|4|64.9|
|llama3.1:8b|BERTimbau\_Large|Médio|6|64.2|
|llama3.1:8b|mBERT|Difícil|4|63.2|
|llama3.1:8b|mBERT|Médio|6|62|



## Desempenho por tipo de questão

|**Modelo**|**Modelo BERTScore**|**Tipo Questão Discursiva**|**Quantidade Questões**|**F1-Score (%)**|
|-|-|-|-:|-:|
|gemma3:12b|BERTimbau\_Large|2 itens (A e B)|8|63.9|
|gemma3:12b|BERTimbau\_Large|1 item (Peça profissional)|2|62.8|
|gemma3:12b|mBERT|1 item (Peça profissional)|2|65.4|
|gemma3:12b|mBERT|2 itens (A e B)|8|57.9|
|jurema:7b|BERTimbau\_Large|1 item (Peça profissional)|2|63|
|jurema:7b|BERTimbau\_Large|2 itens (A e B)|8|61.5|
|jurema:7b|mBERT|1 item (Peça profissional)|2|65.7|
|jurema:7b|mBERT|2 itens (A e B)|8|55.6|
|llama3.1:8b|BERTimbau\_Large|1 item (Peça profissional)|2|66.9|
|llama3.1:8b|BERTimbau\_Large|2 itens (A e B)|8|63.9|
|llama3.1:8b|mBERT|1 item (Peça profissional)|2|69.6|
|llama3.1:8b|mBERT|2 itens (A e B)|8|60.7|



## Observações:

* O **BERTScore não substitui a correção analítica oficial da OAB**.
* A coluna "F1-Score (%)" corresponde ao F1 do BERTScore multiplicado por 100.
* A coluna "resposta_referencia_limpa_guidelines" é a resposta do gabarito espelho da OAB sem os blocos de textos que tratam sobre a pontuação e valores, a fim de haver uma comparação semântica automática mais justa.
* As respostas foram coletadas sob "temperatura 0" para minimizar alucinação.



## Gráficos gerados

!\[grafico\_media\_f1\_percent\_geral.png](grafico\_media\_f1\_percent\_geral.png)

!\[grafico\_media\_f1\_percent\_por\_area.png](grafico\_media\_f1\_percent\_por\_area.png)

!\[grafico\_media\_f1\_percent\_por\_dificuldade.png](grafico\_media\_f1\_percent\_por\_dificuldade.png)

!\[grafico\_media\_f1\_percent\_por\_tipo\_item.png](grafico\_media\_f1\_percent\_por\_tipo\_item.png)



## DATASET QUESTÕES OBJETIVAS

* Formado por 123 questões da prova do 16º e do 17º Exame da OAB.



# Resumo do desempenho dos modelos nas questões objetivas (123 questões)

* A avaliação abaixo considera a acurácia (percentual de acerto) dos modelos em questões de múltipla escolha dos Exames 16º e 17º da OAB.

| Modelo      |   Total Questões |   Total Acertos |   Acurácia (%) |
|:------------|-----------------:|----------------:|---------------:|
| jurema:7b   |              123 |              40 |          32.52 |
| llama3.1:8b |              123 |              60 |          48.78 |
| gemma3:12b  |              123 |              68 |          55.28 |


## Observação:

* As respostas foram coletadas via Ollama API usando "format: json" e "temperatura 0" para garantir a integridade da extração da letra da alternativa assinalada.

