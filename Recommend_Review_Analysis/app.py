from flask import Flask, render_template, request, flash, redirect, url_for
from flask_debug import Debug
from werkzeug.utils import secure_filename

#lib for face Recognation
import os
import re
import ftfy
import nltk
import smtplib
import random
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nltk import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
import re

from sklearn.feature_extraction.text import TfidfTransformer

from collections import Counter

app = Flask(__name__)
app.secret_key = "secret key"

input_dir = 'input'
model_dir = 'model'
model_str_file = 'model_structure.json'
model_weights_file = 'model_weights.h5'
cword_file = 'cword_dict.pkl'
tokenizer_file = 'tokens.pkl'

class_names = [
     'neg',
    'pos'   
]
print(class_names)

cList = pkl.load(open(os.path.join(input_dir,cword_file),'rb'))

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

trained_tokenizer = pkl.load(open(os.path.join(model_dir,tokenizer_file),'rb'))

def load_trained_model(model_str_path, model_wt_path):
    f = Path(model_str_path)
    model_structure = f.read_text()
    trained_model = model_from_json(model_structure)
    trained_model.load_weights(model_wt_path)
    return trained_model

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

def clean_review(reviews):
    cleaned_review = []
    for review in reviews:
        review = str(review)
        if re.match("(\w+:\/\/\S+)", review) == None:
            review = ' '.join(re.sub("(@[A-Za-z]+)|(\#[A-Za-z]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", review).split())
            review = ftfy.fix_text(review)
            review = expandContractions(review)
            review = ' '.join(re.sub("([^A-Za-z \t])", " ", review).split())
            stop_words = stopwords.words('english')
            word_tokens = nltk.word_tokenize(review) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            review = ' '.join(filtered_sentence)
            #review = PorterStemmer().stem(review)
            cleaned_review.append(review.lower())
    return cleaned_review


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


@app.route("/")
def index():
    return render_template("index1.html")


def predict_result(data,text):
    p_review = []
    n_review = []
    str_path = os.path.join(model_dir,model_str_file)
    wt_path = os.path.join(model_dir,model_weights_file)

    model = load_trained_model(str_path, wt_path)

    result = model.predict(data)
    Y_pred = np.round(result.flatten())
    print(Y_pred)
    for i in range(len(Y_pred)):
        print("Text : ",text[i])
        print("Result : ",class_names[int(Y_pred[i])])
        m = class_names[int(Y_pred[i])]
        if m == 'pos':
            p_review.append(text[i])
        else:
            n_review.append(text[i])
    return p_review,n_review

@app.route('/predict' ,methods=['POST','GET'])
def predict():

    p = ['Geneva Platinum Silicone Strap Analogue Watch for Women & Girls - GP-379',
       "Geneva Platinum Analogue Gold Dial Women's Watch -GNV01",
       "IIk Collection Watches Stainless Steel Chain Day and Date Analogue Silver Dial Women's Watch",
       "NUBELA Analogue Prizam Glass Black Dial Girl's Watch",
       'Analogue Pink',
       "Sonata Analog Champagne Dial Women's Watch-NK87018YM01",
       "Sonata Analog White Dial Women's Watch -NJ8989PP03C",
       "Everyday Analog Black Dial Women's Watch -NK8085SM01",
       "Sonata SFAL Analog Silver Dial Women's Watch -NK8080SM01",
       "Timewear Analogue Round Beige Dial Women's Watch - 107Wdtl",
       "TIMEWEAR Analogue Brown Dial Women's Watch - 134Bdtl",
       "Sonata SFAL Analog Silver Dial Women's Watch -NK8080SM-3372",
       "Adamo Analog Blue Dial Women's Watch-9710SM01",
       "ADAMO Aritocrat Women's & Girl's Watch BG-335",
       "Imperious Analog Women's Watch 1021-1031",
       "IIK Collection Watches Analogue Silver Dial Girl's & Women's Analogue Watch - IIK-1033W",
       "Sonata Analog White Dial Women's Watch -NJ8976SM01W",
       "Geneva Platinum Analogue Rose Gold Dial Women'S Watch- Gp-649",
       "Geneva Platinum Analogue Rose Gold Dial Women's Watch- Gp-649",
       "A R Sales Analogue Girl's and Women's Watch",
       "Foxter Bangel Analog White Dial Women's Watch",
       'Howdy Women Watch']
    prodect = []
    if request.method == 'POST':
        prodect.append(int((request.form['q2'])))

    print(prodect)
    df1 = pd.read_csv("input/Final_reviews.csv")

    # df1['product_name'].unique()
    p_name = p[prodect[0]-1]


    core_review = []
    
    for i,j in df1.iterrows():
        if j['product_name'] == p_name:
            # if j['Experience'] >= 5 and j['Helpful_Votes'] >=20 and j['Purchase'] == 'Verified Purchase':
            core_review.append(j['Text'])   

    cleaned_text = clean_review(core_review)

    sequences_text_token = trained_tokenizer.texts_to_sequences(cleaned_text)

    data = pad_sequences(sequences_text_token, maxlen=140)
    print(data)

    # p_cleaned_text = clean_review(p_review)
    # n_cleaned_text = clean_review(n_review)

    p_review,n_review = predict_result(data=data,text=cleaned_text)

    
    p_re = random.sample(p_review,3)
    # p_keys = random.sample(p_key,3)
    n_re = random.sample(n_review,3)
    # n_keys = random.sample(n_key,3)

     
    # res = [p_name,p_re,p_keys,n_re,n_keys]
        
    print(p_re,'\n',n_re)

    return render_template("result1.html" , p_name = p_name,p_re = p_re, n_re=n_re)



if __name__ == '__main__':
    app.run(debug=True,threaded=False)