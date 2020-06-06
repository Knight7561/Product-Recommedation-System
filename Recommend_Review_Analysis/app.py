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

import math
# import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

app = Flask(__name__)
app.secret_key = "secret key"

input_dir = 'input'
model_dir = 'model'
model_str_file = 'model_structure.json'
model_weights_file = 'model_weights.h5'
cword_file = 'cword_dict.pkl'
tokenizer_file = 'tokens.pkl'
review_file="input/Final_reviews.csv"
product_category="Watches"
model_str_file_camera = 'model_structure_camera.json'
model_weights_file_camera = 'model_weights_camera.h5'
cword_file = 'cword_dict.pkl'
tokenizer_file_camera = 'tokensCamera.pkl'
review_file_camera="input/Connected Camera reviews.csv"

class_names = [
     'neg',
    'pos'   
]
print(class_names)

cList = pkl.load(open(os.path.join(input_dir,cword_file),'rb'))

c_re = re.compile('(%s)' % '|'.join(cList.keys()))



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


# def sort_coo(coo_matrix):
#     tuples = zip(coo_matrix.col, coo_matrix.data)
#     return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
# def extract_topn_from_vector(feature_names, sorted_items, topn=10):
#     """get the feature names and tf-idf score of top n items"""
    
#     #use only topn items from vector
#     sorted_items = sorted_items[:topn]
 
#     score_vals = []
#     feature_vals = []
    
#     # word index and corresponding tf-idf score
#     for idx, score in sorted_items:
        
#         #keep track of feature name and its corresponding score
#         score_vals.append(round(score, 3))
#         feature_vals.append(feature_names[idx])
 
#     #create a tuples of feature,score
#     #results = zip(feature_vals,score_vals)
#     results= {}
#     for idx in range(len(feature_vals)):
#         results[feature_vals[idx]]=score_vals[idx]
    
#     return results


def recommend_items(userID, pivot_df, preds_df, num_recommendations):
    # index starts at 0  
    user_idx = userID-1 
    # Get and sort the user's ratings
    sorted_user_ratings = pivot_df.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_ratings
    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_predictions
    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1,sort=False)
       
    temp.index.name = 'Recommended Items'
    temp.columns = ['user_ratings', 'user_predictions']
    temp = temp.loc[temp.user_ratings == 0]   
    temp = temp.sort_values('user_predictions', ascending=False)
    print(temp.head())
    return temp.head(num_recommendations)

def recom(userID,seller):
    if seller == 1:
        user_df=pd.read_excel('input/celler_1.xlsx')
    elif seller == 2:
        user_df=pd.read_excel('input/celler_2.xlsx')
    
    print(user_df.head())   
    user_df = user_df.dropna()
    counts = user_df.user.value_counts()
    #print("counts is",counts.index)
    user_df_final = user_df[user_df.user.isin(counts[counts>=2].index)]
    train_data = user_df_final
    user_df_CF = train_data
    pivot_df = user_df_CF.pivot_table(index = 'user', columns ='product_code', values = 'rating').fillna(0)
    #print("pivot_df is \n",pivot_df)
    pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)
    #print("pivot_df after  is \n",pivot_df)
    pivot_df.set_index(['user_index'], inplace=True)
    #print("pivot_df after setting is \n",pivot_df)
    U, sigma, Vt = svds(pivot_df, k = 8)
    #print("U is ",U)
    #print("sigma is ",sigma)
    #print("Vt is ",Vt)
    sigma = np.diag(sigma)

    #print("sigma diagonal matrix is",sigma)
    #Predicted ratings
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
    #print("all_user_predicted_ratings",all_user_predicted_ratings)

    # Convert predicted ratings to dataframe
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = pivot_df.columns)
    #print("preds_df is ",preds_df)
    a = recommend_items(userID, pivot_df, preds_df, 5)
    #print("a is ",a)
    return a
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################

@app.route("/")
def index():
    return render_template("indexHome.html")

@app.route("/customer")
def indexCustomer():
    return render_template("indexCustomer.html")

@app.route("/seller")
def indexSeller():
    return render_template("sellerSelect.html")


def predict_result(data,text):
    p_review = []
    n_review = []
    str_path = os.path.join(model_dir,model_str_file)
    wt_path = os.path.join(model_dir,model_weights_file)
    model = load_trained_model(str_path, wt_path)
    result = model.predict(data)
    #print("Predicted result without flattening is ",result)
    Y_pred = np.round(result.flatten())
    #print(Y_pred)
    for i in range(len(Y_pred)):
        #DisableCommentprint("Text : ",text[i])
        #DisableCommentprint("Result : ",class_names[int(Y_pred[i])])
        m = class_names[int(Y_pred[i])]
        if m == 'pos':
            p_review.append(text[i])
        else:
            n_review.append(text[i])
    return p_review,n_review

@app.route('/res' ,methods=['POST','GET'])
def res():
    global review_file
    product = []
    if request.method == 'POST':
        product.append(request.form['q1'])
    print("Product selected is ",product)
    p_name = product[0]
    #print(p_name)
    core_review = []
    #print("review file path is ",review_file)
    df1 = pd.read_csv(review_file)###########
    #print("df1 is ",df1)
    for i,j in df1.iterrows():
        if j['product_name'] == p_name:
            # if j['Experience'] >= 5 and j['Helpful_Votes'] >=20 and j['Purchase'] == 'Verified Purchase':
            core_review.append(j['Text'])   

    #print("Core Reviews before cleaning is",core_review)
    cleaned_text = clean_review(core_review)
    #print("Core Reviews After cleaning is",cleaned_text)

    trained_tokenizer = pkl.load(open(os.path.join(model_dir,tokenizer_file),'rb'))
    sequences_text_token = trained_tokenizer.texts_to_sequences(cleaned_text)#################
    #print("sequences_text_token is",sequences_text_token)

    data = pad_sequences(sequences_text_token, maxlen=140)
    #print("data is ",data,"   dd")
    #DISABLE COMMENTprint(data)

    # p_cleaned_text = clean_review(p_review)
    # n_cleaned_text = clean_review(n_review)

    p_review,n_review = predict_result(data=data,text=cleaned_text)

    #print("len of positive is",len(p_review),"len of negetive is",len(n_review))
    p_re = random.sample(p_review,5)
    # p_keys = random.sample(p_key,3)
    n_re = random.sample(n_review,5)
    # n_keys = random.sample(n_key,3)

     
    # res = [p_name,p_re,p_keys,n_re,n_keys]
        
    #DISABLE COMMENTprint(p_re,'\n',n_re)

    #Collection of metadata about the product
    df_metadata_watch=pd.read_csv("input/watch_analystics_final.csv")
    for e,i in df_metadata_watch.iterrows():
        if(i['Product_name']==p_name):
            metadata={}
            metadata['p_name']=i["Product_name"]
            metadata["total_no_reviews"]=i["Total"]
            metadata["rate_1"]=i["1_star"]
            metadata["rate_2"]=i["2_star"]
            metadata["rate_3"]=i["3_star"]
            metadata["rate_4"]=i["4_star"]
            metadata["rate_5"]=i["5_star"]
            metadata["imgLink"]=i["Image_Link"]
    print("\n\n\n\n\nmetadata is",metadata)
    return render_template("result1Customer.html",p_name=p_name,p_re=p_re,n_re=n_re,p_re_len=len(p_review),n_re_len=len(n_review),metadata=metadata)

@app.route('/predictCustomer' ,methods=['POST','GET'])
def predictCustomer():
    print("q3 is ",request.form['q3'])
    if((int(request.form['q3'])==2)):
        global review_file
        product_category="Camera"
        model_str_file=model_str_file_camera
        model_weights_file=model_weights_file_camera
        cword_file = 'cword_dict.pkl'
        tokenizer_file=tokenizer_file_camera
        review_file=review_file_camera
        print("Items to be stocked at ",review_file)
        p=['Canon SX50', 'Canon T5 Bundle v2',
       'Nikon COOLPIX AW100 16 MP CMOS Waterproof Digital Camera with GPS and Full HD 1080p Video',
       'Nikon COOLPIX AW110 16 MP Waterproof Digital Camera with Built-In Wi-Fi',
       'Nikon COOLPIX S3600 20.1 MP Digital Camera with 8x Zoom NIKKOR Lens and 720p HD Video',
       'Panasonic  Digital Camera with 3-Inch LCD',
       'Panasonic Lumix 12.1 MP Digital Camera with CMOS Sensor and 24x Optical Zoom - Black',
       'Panasonic LUMIX DMC-FZ70 16.1 MP Digital Camera with 60x Optical Image Stabilized Zoom and 3-Inch LCD (Black)',
       'Panasonic LUMIX DMC-LX7K 10.1 MP Digital Camera with 3.8x Optical zoom and 3.0-inch LCD -  Black',
       'Panasonic Lumix DMC-TS25 16.1 MP Tough Digital Camera with 8x Intelligent Zoom',
       'Samsung SmartCam HD Pro 1080p Full-HD Wi-Fi Camera',
       'Samsung SNH-1011 Wireless IP Camera',
       'Samsung WB1100F 16.2MP CMOS Smart WiFi & NFC Digital Camera with 35x Optical Zoom, 3.0" LCD and 1080p HD Video',
       'Sony Alpha a5000 Mirrorless Digital Camera with 16-50mm OSS Lens (White)',
       'Nikon COOLPIX AW120', 'Nikon Coolpix L330 Digital Camera (Black)',
       'Nikon COOLPIX L810 16.1 MP Digital Camera with 26x Zoom NIKKOR ED Glass Lens and 3-inch LCD',
       'Nikon COOLPIX L820 16 MP Digital Camera with 30x Zoom',
       'Nikon COOLPIX P520 18.1 MP Digital Camera with 42x Zoom',
       'Nikon COOLPIX S32',
       'Nikon COOLPIX S3500 20.1MP Digital Camera with 7x Zoom',
       'Sony Alpha a6000 Mirrorless Digital Camera with 16-50mm Power Zoom Lens']
        product = []
        cell = []
        if request.method == 'POST':
            product.append(int(request.form['q1']))
            cell.append(int(request.form['q2']))
        print(product[0],cell[0])
        recomen_pro = recom(product[0],cell[0])
        l=recomen_pro.index[0:]
        res = list(l)
        # print("res",res)
        # res
        names = []
        for i in res:
            names.append(p[i])
        # print("names is: ",names)
        # df1['product_name'].unique()
    else:
        product_category="Watches"
        model_str_file = 'model_structure.json'
        model_weights_file = 'model_weights.h5'
        cword_file = 'cword_dict.pkl'
        tokenizer_file = 'tokens.pkl'
        review_file="input/Final_reviews.csv"
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
        product = []
        cell = []
        if request.method == 'POST':
            product.append(int(request.form['q1']))
            cell.append(int(request.form['q2']))

        print(product[0],cell[0])
        recomen_pro = recom(product[0],cell[0])
        # a = recom(1,0)
        # print("recomen_pro",recomen_pro)
        
        l=recomen_pro.index[0:]
        res = list(l)
        # print("res",res)
        # res
        names = []
        for i in res:
            names.append(p[i])
        # print("names is: ",names)
        # df1['product_name'].unique()
        c1 = ["NUBELA Analogue Prizam Glass Black Dial Girl's Watch",
        'Analogue Pink',
        "Sonata Analog Champagne Dial Women's Watch-NK87018YM01",
        "Sonata Analog White Dial Women's Watch -NJ8989PP03C",
        "Everyday Analog Black Dial Women's Watch -NK8085SM01"]
        c2 = ["Sonata Analog White Dial Women's Watch -NJ8976SM01W",
        "Geneva Platinum Analogue Rose Gold Dial Women'S Watch- Gp-649",
        "Geneva Platinum Analogue Rose Gold Dial Women's Watch- Gp-649",
        "A R Sales Analogue Girl's and Women's Watch",
        "Foxter Bangel Analog White Dial Women's Watch"]
        if len(names) <= 0 and cell[0] == 1:
            for i in c1:
                names.append(i)
        elif len(names) <= 0 and cell[0] == 2:
            for i in c2:
                names.append(i)
        # print("names after apend is",names)
    return render_template("recomCustomer.html" , p_name = names)


@app.route('/sellerWatch' ,methods=['POST','GET'])
def sellerWatch():
    id=(int((request.args['id'])))
    df_items=pd.read_csv("input/Final_reviews.csv")
    items_numbers=df_items.product_name.value_counts().rename_axis('Product names').reset_index(name='Count')
    print("items_numbers ",items_numbers)
    if(id==1):
        return render_template("indexSeller.html" , productname=items_numbers['Product names'][0:11],reviewcount=items_numbers['Count'][0:11])
    elif(id==2):
        return render_template("indexSeller.html" , productname=items_numbers['Product names'][11:],reviewcount=items_numbers['Count'][11:])

@app.route('/sellerCamera' ,methods=['POST','GET'])
def sellerCamera():
    id=(int((request.args['id'])))
    df_items=pd.read_csv("input/Connected Camera reviews.csv")
    items_numbers=df_items.product_name.value_counts().rename_axis('Product names').reset_index(name='Count')
    # print("items_numbers ",items_numbers['Product names'][11:])
    if(id==1):
        return render_template("indexSeller.html" , productname=items_numbers['Product names'][0:11],reviewcount=items_numbers['Count'][0:11])
    elif(id==2):
        return render_template("indexSeller.html" , productname=items_numbers['Product names'][11:],reviewcount=items_numbers['Count'][11:])


@app.route('/predictSeller' ,methods=['POST','GET'])
def predictSeller():
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
    product = []
    if request.method == 'POST':
        product.append(int((request.form['q2'])))
    if request.method == 'GET':
        product.append(int((request.args['q2'])))

    print(product)
    df1 = pd.read_csv("input/Final_reviews.csv")

    # df1['product_name'].unique()
    p_name = p[product[0]-1]


    core_review = []
    
    for i,j in df1.iterrows():
        if j['product_name'] == p_name:
            # if j['Experience'] >= 5 and j['Helpful_Votes'] >=20 and j['Purchase'] == 'Verified Purchase':
            core_review.append(j['Text'])   

    cleaned_text = clean_review(core_review)

    trained_tokenizer = pkl.load(open(os.path.join(model_dir,tokenizer_file),'rb'))
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

    return render_template("result1Seller.html" , p_name = p_name,p_re = p_re, n_re=n_re)

if __name__ == '__main__':
    app.run(debug=True,threaded=False)