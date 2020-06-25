#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return (black_friday.shape[0], black_friday.shape[1])


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


mask_gender = black_friday.loc[:,'Gender'] == 'F'
mask_age = black_friday.loc[:, 'Age'] == '26-35'

black_friday.loc[mask_gender & mask_age].loc[:,'Gender'].count()


# In[6]:


def q2():
    
    mask_gender = black_friday.loc[:,'Gender'] == 'F'
    mask_age = black_friday.loc[:, 'Age'] == '26-35'
    female_26to35 = black_friday.loc[mask_gender & mask_age].loc[:,'Gender'].count()
    # Retorne aqui o resultado da questão 2.
    
    return int(female_26to35)


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[12]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[14]:


def q4():
    
    # Retorne aqui o resultado da questão 4.
    
    return len(black_friday.dtypes.unique())


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[36]:


def q5():
    
    # Retorne aqui o resultado da questão 5.
    
    return float(1 - len(black_friday.dropna()) / len(black_friday))


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[9]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return int(black_friday.isna().sum().sort_values(ascending=False)[0])


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[43]:


mask_not_na = black_friday.loc[:, 'Product_Category_3'].notna()

black_friday.loc[mask_not_na].loc[:, 'Product_Category_3'].value_counts().index[0]


# In[10]:


def q7():
    # Retorne aqui o resultado da questão 7.
    mask_not_na = black_friday.loc[:, 'Product_Category_3'].notna()

    return black_friday.loc[mask_not_na].loc[:, 'Product_Category_3'].value_counts().index[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[33]:


def normalizer(df, column):
    
    x_min = df[column].min()
    x_max = df[column].max()
    
    return (df[column] - x_min)/(x_max - x_min)


# In[34]:


float(normalizer(black_friday, 'Purchase').mean())


# In[11]:


def q8():
    
    # Retorne aqui o resultado da questão 8.
    normalized = normalizer(black_friday, 'Purchase')
    
    return float(normalized.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[44]:


def padronizer(df, column):
    
    mu = df[column].mean()
    sigma = df[column].std()
    
    return (df[column] - mu)/(sigma)


# In[45]:


padronized = padronizer(black_friday, 'Purchase')

((-1 <= padronized ) & ( 1 >= padronized )).sum()


# In[43]:


def q9():
    
    # Retorne aqui o resultado da questão 9.
    padronized = padronizer(black_friday, 'Purchase')
    between_sigma = ((-1 <= padronized ) & ( 1 >= padronized )).sum()
    return int(between_sigma)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[51]:


# Dos que são 'na' na coluna 'Product_Category_2',
# quantos não são 'na' na coluna 'Product_Category_3'?
black_friday[black_friday['Product_Category_2'].isna()]['Product_Category_3'].notna().sum()

# Se o número for igual a 0, todos os 'na' na primeira coluna também o são na segunda e teremos True
# Caso contrário, False


# In[59]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return bool(black_friday[black_friday['Product_Category_2'].isna()]['Product_Category_3'].notna().sum() == 0)

