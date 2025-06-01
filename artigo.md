# IA Literária: Um Sistema Inteligente para Classificação de Livros por Gênero

Este projeto tem como objetivo aplicar Inteligência Artificial ao domínio de documentos textuais, utilizando exclusivamente bibliotecas Python para construir um sistema simples e eficiente de classificação automática de livros por gênero literário. A proposta faz parte da temática "IA Aplicada a Documentos" e trata obras literárias — como sinopses, resumos e trechos — como documentos a serem analisados por modelos de linguagem.

## **Resumo**
O projeto tem como objetivo construir um modelo de aprendizado de máquina capaz de classificar livros em gêneros literários a partir de suas descrições textuais. A classificação automática de gêneros visa auxiliar plataformas literárias, sistemas de recomendação e bibliotecas digitais, tornando a organização e o acesso ao conteúdo mais eficientes.

A base de dados original foi obtida a partir de um dataset público disponível no Kaggle, contendo cerca de 10 mil registros de livros com informações como título, descrição e lista de gêneros associados. A partir dessa base bruta, foi criada a BASE_MODELO, com os campos renomeados e ajustados para facilitar o processamento. A lista de gêneros, originalmente armazenada como string de listas, foi convertida para listas reais, permitindo a expansão de múltiplos gêneros por livro em diferentes linhas do dataset.

Para o pré-processamento textual, foram utilizados recursos da biblioteca NLTK, como remoção de pontuação, stopwords, lematização e tokenização. A vetorização dos textos foi realizada com TfidfVectorizer do scikit-learn, configurado com n-gramas até 5 palavras e limite de frequências para filtrar ruído textual. O modelo de classificação escolhido foi a Regressão Logística, configurada com class_weight="balanced" para lidar com o desbalanceamento entre gêneros. A base foi dividida estratificadamente em treino e teste, e a performance foi avaliada com acurácia e métricas detalhadas (precision, recall, F1-score).

Após diversas iterações e melhorias no pré-processamento, o modelo atingiu uma acurácia de aproximadamente 65,1% ao classificar livros entre os 10 gêneros mais frequentes. O desempenho variou de acordo com a quantidade de exemplos por classe, com maior precisão em categorias amplamente representadas, como Fiction, Fantasy e Nonfiction. A estratégia de expansão dos gêneros por linha e a escolha cuidadosa dos parâmetros de vetorização contribuíram significativamente para a melhoria da performance geral.

## **Preparação e transformação dos dados**

A etapa de preparação dos dados foi fundamental para garantir a qualidade do modelo de classificação. Inicialmente, os dados foram importados a partir de um arquivo .xlsx previamente estruturado com base em um dataset original do Kaggle. Este arquivo continha aproximadamente 10 mil registros de livros, com informações como título, descrição e uma lista de gêneros atribuídos a cada obra.

Uma das primeiras ações de limpeza foi eliminar entradas que não possuíam descrição textual e aquelas em que a lista de gêneros estava vazia. Em seguida, a coluna que armazenava os gêneros — originalmente no formato de string representando listas — foi convertida para listas reais utilizando ast.literal_eval(). Essa transformação possibilitou o uso do método explode() do pandas, permitindo que cada gênero de um livro fosse tratado como uma instância separada, ou seja, um mesmo livro poderia aparecer mais de uma vez no conjunto de dados, cada linha associada a um de seus gêneros. Isso garantiu maior granularidade para a tarefa de classificação.

Após essa expansão, os textos das descrições passaram por um rigoroso pré-processamento textual. Essa etapa incluiu a remoção de números, pontuações e palavras irrelevantes (stopwords), além da normalização das palavras por meio da lematização — técnica que reduz palavras às suas formas base. Essa transformação foi realizada utilizando bibliotecas da NLTK, como WordNetLemmatizer e stopwords.

Com os textos processados, foi realizada a vetorização utilizando o método TF-IDF (Term Frequency-Inverse Document Frequency), que transforma os textos em representações numéricas baseadas na frequência e relevância dos termos. O vetor foi configurado para extrair n-gramas de 1 a 5 palavras, com um limite de palavras muito comuns e muito raras para evitar ruído e overfitting.

Por fim, foram selecionados os 10 gêneros mais frequentes no dataset, e o dataframe foi filtrado para manter apenas os registros pertencentes a essas categorias. Essa decisão teve como objetivo equilibrar o modelo e melhorar a capacidade preditiva, evitando o viés causado por classes pouco representadas. Também foi garantido que apenas um registro por livro fosse mantido para evitar duplicidade de informações na base final utilizada para treino e teste.

## **Fine Tuning do modelo**
Após a construção do pipeline inicial de classificação de gêneros literários com base na descrição dos livros, foi conduzido um processo de fine tuning do modelo com o objetivo de maximizar a acurácia e melhorar o equilíbrio entre precisão e recall para as principais classes. A estratégia de ajuste fino concentrou-se em três frentes principais: seleção de atributos, balanceamento das classes e otimização dos hiperparâmetros do algoritmo.

A primeira etapa consistiu em aprimorar o processo de vetorização textual com o uso do TfidfVectorizer. Foram testadas diferentes configurações de n-gramas (de unigramas até pentagramas) e valores de corte para min_df e max_df, que controlam a inclusão de termos com baixa ou alta frequência no vocabulário. Esse ajuste teve como objetivo encontrar um equilíbrio entre a expressividade dos textos e o ruído causado por termos irrelevantes ou extremamente específicos.

Em seguida, foi observada uma grande variação na distribuição das classes, ou seja, alguns gêneros estavam muito mais representados do que outros. Para lidar com esse desbalanceamento, foi utilizado o parâmetro class_weight='balanced' no algoritmo de regressão logística, o qual ajusta automaticamente os pesos das classes com base na frequência relativa de cada uma. Essa técnica ajudou o modelo a tratar com mais equidade os gêneros menos frequentes no conjunto de dados.

Por fim, foram realizados testes com diferentes hiperparâmetros do modelo LogisticRegression, como o número máximo de iterações (max_iter) e o parâmetro de regularização C, responsável por controlar o overfitting. O valor de C=1.0 apresentou um bom equilíbrio entre complexidade e generalização, e max_iter foi aumentado para 1000 para garantir a convergência do algoritmo.

O resultado desse processo de fine tuning foi um aumento consistente na acurácia do modelo, atingindo cerca de 65,1%, com melhora significativa nos scores de F1 em várias classes. Apesar de ainda haver desafios no desempenho de gêneros com baixa representatividade, o modelo demonstrou maior robustez e capacidade preditiva após os ajustes finos aplicados.

## **Avaliação do modelo**

A avaliação do modelo de classificação foi conduzida utilizando um conjunto de teste separado (20% dos dados), garantindo uma análise imparcial da performance preditiva da solução. As métricas utilizadas foram acurácia, precisão, recall e F1-score, fornecendo uma visão abrangente do desempenho, especialmente considerando o desequilíbrio entre os gêneros literários.

O modelo final, baseado em regressão logística com vetorização TF-IDF, atingiu uma acurácia global de aproximadamente 65,1%. Embora a acurácia seja uma métrica relevante, ela pode ser enganosa em cenários com classes desbalanceadas, como é o caso de alguns gêneros menos frequentes. Por isso, foi fundamental observar o comportamento do modelo em métricas como o F1-score, que combina precisão e recall, permitindo avaliar o equilíbrio entre falsas classificações positivas e negativas.

A análise do relatório de classificação revelou que os gêneros mais bem representados, como Fiction, Fantasy e Nonfiction, obtiveram F1-scores mais altos, indicando que o modelo conseguiu aprender padrões consistentes nessas categorias. Por outro lado, gêneros com menos amostras, como Contemporary ou Audiobook, apresentaram desempenho inferior, com métricas próximas de zero, refletindo a dificuldade do modelo em generalizar quando há pouca informação disponível para aprendizado.

Além disso, o uso de class_weight='balanced' na regressão logística contribuiu para mitigar parcialmente os efeitos do desbalanceamento, permitindo que o modelo prestasse mais atenção às classes menos frequentes. No entanto, o fato de algumas classes ainda não terem nenhuma amostra corretamente prevista indica a necessidade de estratégias adicionais, como aumento de dados (data augmentation), reamostragem (oversampling/undersampling), ou mudança para modelos mais sofisticados como redes neurais ou transformers.

Em suma, o modelo apresentou desempenho satisfatório dado o cenário e as limitações dos dados. A acurácia de 65,1% e os F1-scores médios por classe indicam que a abordagem tem potencial prático, especialmente para os gêneros mais frequentes, ao mesmo tempo em que destaca oportunidades de melhoria para uma cobertura mais equilibrada de todos os gêneros literários.

## **Consumo do modelo**

Após o treinamento e validação, o modelo foi integrado a uma função de inferência chamada classificar_livro, que permite prever o gênero literário predominante de um livro com base apenas em sua descrição textual. Esse consumo do modelo foi estruturado de forma simples, eficiente e acessível, tornando possível sua incorporação em aplicações reais, como sistemas de recomendação, bibliotecas digitais ou plataformas de e-commerce de livros.

O fluxo de uso segue três etapas principais: primeiro, o texto da descrição do livro é submetido ao mesmo processo de pré-processamento aplicado durante o treinamento (remoção de números, pontuações, stopwords e lematização). Em seguida, o texto limpo é vetorizado utilizando o modelo TF-IDF previamente treinado. Por fim, a representação vetorial é passada ao classificador de regressão logística, que retorna a previsão de gênero.

Esse modelo pode é consumido diretamente via interface Streamlit. Nessa interface, o usuário pode escolher um livro e visualizar instantaneamente a classificação do gênero sugerida pelo modelo. Esse consumo em tempo real amplia significativamente as possibilidades de aplicação da solução, tornando-a adequada tanto para fins educacionais quanto comerciais.

A leveza do pipeline — com TF-IDF e regressão logística — também garante rápida resposta e baixa demanda computacional, facilitando o uso do modelo mesmo em ambientes com recursos limitados. Com isso, o modelo se torna não apenas funcional e preciso, mas também acessível para diferentes tipos de usuários e plataformas.