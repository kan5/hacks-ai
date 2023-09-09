from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

import json
import random
import string
import pandas
from ast import literal_eval
import os


import pandas as pd
import nltk
# nltk.download('stopwords')
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import spacy
from sklearn.cluster import KMeans
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


#input: series
#output: df cluster_id, inputs, label
# survey = {id, name, [{question}, ]}
# sessions = dict()


import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True)


ready_json = dict()
surveys = []

ready_surveys = {}

app = FastAPI()

stopwords_russian = stopwords.words('russian')
stopwords_english = stopwords.words('english')
all_stopwords = list(set(stopwords_russian + stopwords_english))
nlp = spacy.load("ru_core_news_sm")
def run_model(text_list: list):
    text_series = pd.Series(text_list, name='text')
    def clean(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Zа-яА-Я ]+", "", text).strip()
        return text
    
    def lemmatize_spacy(text):
        return " ".join([token.lemma_ for token in nlp(text)])
    corpus = text_series.apply(clean).apply(lemmatize_spacy)
    
    tf_idf = TfidfVectorizer(stop_words=all_stopwords)
    corpus_vec = tf_idf.fit_transform(corpus)
    agg_clustering = AgglomerativeClustering(n_clusters=8)
    clusters = agg_clustering.fit_predict(corpus_vec.toarray())
    # функция для определения топ 5 слов в кластере
    def top_tfidf_words(cluster_label, tfidf_matrix, feature_names, top_n=5):
        cluster_indices = np.where(clusters == cluster_label)[0]
        cluster_tfidf_scores = tfidf_matrix[cluster_indices].sum(axis=0).A1
        top_indices = np.argsort(cluster_tfidf_scores)[::-1][:top_n]
        top_words = [feature_names[i] for i in top_indices]
        return top_words
    data = []
    feature_names = tf_idf.get_feature_names_out()
    for cluster_label in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_label)[0]
        cluster_sentences = text_series.iloc[cluster_indices].tolist()
        top_words = top_tfidf_words(cluster_label, corpus_vec, feature_names)
        
        data.append({
            'answers': cluster_sentences,
            'name': top_words
        })
    df = pd.DataFrame(data)
    return df
    

def run_sentiment(text: str):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    return int(predicted[0])


@app.post("/upload_survey/")
async def upload_survey(file: Annotated[bytes, File()]):
    global surveys
    survey = literal_eval(file.decode('utf-8-sig'))
    surveys.append(survey)
    return {"file_size": len(file)}


@app.post("/upload_question/{survey_id}")
async def upload_question(survey_id: int, file: UploadFile):
    global surveys
    for sv in surveys:
        if sv["id"] == survey_id:
            sv["questoins"].append(literal_eval(file.decode('utf-8-sig')))
    return {"file_size": len(file)}


@app.get("/surveys/")
async def get_surveys():
    global surveys
    return [{"id": sv["id"], "name": sv["name"]} for sv in surveys]


@app.get("/questions/{survey_id}")
async def questions(survey_id: int, has_sentiment: bool = False):
    global surveys
    questions = [] 
    ret = []
    for sv in surveys:
        if sv["id"] == survey_id:
            questions = sv["questions"]
            sv_name = sv["name"]
            break
    for question in questions:
        answers = question["answers"]
        pure_answers = [i["answer"] for i in answers]
        df_clust = run_model(pure_answers)
        df_clust["name"] = df_clust["name"].map(lambda x: ", ".join(x))
        df_clust["answers"] = df_clust["answers"].map(lambda x: [{"text": i} for i in x])
        clusters = df_clust.to_dict('records')
        for i in clusters:
            for j in i["answers"]:
                for u in answers:
                    if u["answer"] == j["text"]:
                        j["counter"] = u["count"]
                        break
                if has_sentiment:
                    j["tonality"] = run_sentiment(j["text"])
        ret.append({"name": question["question"],
                    "clusters": clusters})
    ready_surveys.append({"id": survey_id, 
                          "name": sv_name,
                          "questions": ret})
    return ret


@app.get("/")
async def main():
    content = """
<body>
<form action="/upload_survey/" enctype="multipart/form-data" method="post">
<input name="upload_survey" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)