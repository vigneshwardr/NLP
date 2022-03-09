from bs4 import BeautifulSoup as bs
import requests
import re
import pandas as pd
import matplotlib as plt
from sklearn.metrics import accuracy_score, classification_report

#SQL and Azure connections 
import pyodbc
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential


#general packages 
from pprint import pprint
from time import time
import logging


#importing dataset
from sklearn.datasets import fetch_20newsgroups

#importing feature extraction and transformation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#importing models 
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# importing process flows
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


## connecting to Azure Key vault
KVUri = f"https://bigdataproject.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)
retrieved_secret = client.get_secret('bigdata')
retrieved_secret_value = retrieved_secret.value


## Connecting to SQL server that we created in Azure
cxn = pyodbc.connect(
    'Driver=ODBC Driver 17 for SQL Server;'
    'Server=tcp:bigdatasqlservers.database.windows.net,1433;'
    'Database=bigdata_sqldatabase;'
    'Uid=bigdata;'
    'Pwd={retrieved_secret_value};'
    'Encrypt=yes;'
    'TrustServerCertificate=no;'
    'Connection Timeout=30;'.format(retrieved_secret_value=retrieved_secret_value))


page = 'https://www.yelp.ca/biz/kinka-izakaya-original-toronto?start='
link = page
r = requests.get(link)
r = r.text
soup = bs(r, 'html.parser')
no_of_pages = soup.find_all("span", class_ = "css-1fdy0l5")[0].text
no_of_pages = int(round(int(no_of_pages[:4]),0))-20

def get_datas(page, value):
    link = page+value
    print(link)
    r = requests.get(link)
    r = r.text
    soup = bs(r, 'html.parser')
    print("no_of_text"+str(len(soup.find_all("p", class_ = "comment__09f24__gu0rG css-qgunke"))))
    return soup

## extracting the text and the number of stars
text = []
stars = []


def page_df():    
    for i in range(0,no_of_pages,10):
        i = str(i)
        soup = get_datas(page, i)
        for data in soup.find_all("p", class_ = "comment__09f24__gu0rG css-qgunke"):
            y = data.get_text()
            text.append(y)
        print(len(text))

        datas = soup.find_all("div", class_ = "i-stars__09f24__foihJ")
        for j in range(1,11):
            star = datas[j]['aria-label']
            print(star)
            stars.append(star)
        print(len(stars))
    df = pd.DataFrame({'text':text, 'stars':stars})
    df['stars'] = df['stars'].str.slice(0,1)    
    return df

df = page_df()

##creating cursor object

crsr = cxn.cursor()

# creating and storing data to the azure sql database 
try:
    crsr.execute('drop table raw_scrapped_data')
    crsr.execute('create table raw_scrapped_data (text varchar(max), stars int)')
    for index, row in df.iterrows():
        crsr.execute("INSERT INTO raw_scrapped_data(text,stars) values(?,?)", row.text, row.stars)
    crsr.commit()
except:    
    crsr.execute('create table raw_scrapped_data (text varchar(max), stars int)')
    for index, row in df.iterrows():
            crsr.execute("INSERT INTO raw_scrapped_data(text,stars) values(?,?)", row.text, row.stars)
    crsr.commit()



##machine learning models execution starts
def ml_models():
    clf = ('clf', SGDClassifier(tol=1e-3))
    lr = ('lr', LogisticRegression())
    mnb = ('mnb', MultinomialNB())
    dtc = ('dtc', DecisionTreeClassifier())
    svc = ('svc', SVC()) 
    models = [clf, lr, mnb, dtc, svc]

    pipe = {}

    for model in models:
        pipe[model[0]] = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            model,
        ])


    parameters = {
        'clf': {
    #             'vect__max_df': (0.5, 0.75, 1.0),
            #     'vect__max_features': (None, 5000, 10000, 50000),
    #             'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                'tfidf__use_idf': (True, False),
                'tfidf__norm': ('l1', 'l2'),
                'clf__max_iter': (20,),
            #     'clf__alpha': (0.00001, 0.000001),
                'clf__penalty': ('l2', 'elasticnet'),
            #     'clf__max_iter': (10, 50, 80)
        },
        'lr': {
    #             'vect__max_df': (0.5, 0.75, 1.0),
    #             'vect__max_features': (None, 5000, 10000, 50000),
    #             'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                'tfidf__use_idf': (True, False),
                'tfidf__norm': ('l1', 'l2'),
                'lr__penalty': ('l2','l1', 'elasticnet'),
                'lr__solver': ('sag','saga','newton-cg','lbfgs')
        },
        'mnb': {
    #             'vect__max_df': (0.5, 0.75, 1.0),
    #             'vect__max_features': (None, 5000, 10000, 50000),
    #             'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                'tfidf__use_idf': (True, False),
                'tfidf__norm': ('l1', 'l2')
        },
        'dtc': {
    #             'vect__max_df': (0.5, 0.75, 1.0),
    #             'vect__max_features': (None, 5000, 10000, 50000),
    #             'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                'tfidf__use_idf': (True, False),
                'tfidf__norm': ('l1', 'l2'),
                'dtc__criterion': ('gini','entropy'),
                'dtc__max_depth': (4,5,6)

        },
        'svc': {
    #             'vect__max_df': (0.5, 0.75, 1.0),
    #             'vect__max_features': (None, 5000, 10000, 50000),
    #             'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                'tfidf__use_idf': (True, False),
                'tfidf__norm': ('l1', 'l2'),
                'svc__kernel': ('linear', 'rbf', 'poly')
            
        },
        
    }

    grid = {}
    for model in models:
        print(model[0])
        grid[model[0]] = GridSearchCV(pipe[model[0]], parameters[model[0]], cv=5,
                                n_jobs=-1, verbose=1)
    print(grid)


    grid_scores = {}
    for model in models:
        print(grid[model[0]])
        datat = list(df.text)
        grid[model[0]].fit(datat, df.stars)
        grid_scores[model[0]] = grid[model[0]].best_score_

    
    model_result = pd.DataFrame(grid_scores.items(), columns=['model', 'value'])
    return model_result

model_result = ml_models()

try:
    crsr.execute('drop table model_accuracys')
    crsr.execute('create table model_accuracys (model varchar(255), value float)')
    for index, row in model_result.iterrows():
        crsr.execute("INSERT INTO model_accuracys(model,value) values(?,?)", row.model, row.value)
    crsr.commit()
except:
    crsr.execute('create table model_accuracys (model varchar(255), value float)')
    for index, row in model_result.iterrows():
        crsr.execute("INSERT INTO model_accuracys(model,value) values(?,?)", row.model, row.value)
    crsr.commit()

