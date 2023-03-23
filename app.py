import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import pandas as pd
import tweepy
import config
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from keras.preprocessing.sequence import pad_sequences

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

app = Flask(__name__)
reconstructed_model = keras.models.load_model("sentiment-score-model")
tokenizer= pickle.load(open("tokenizer.pkl", "rb"))
maxlen = 130
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    to_predict_list = request.form.to_dict()

    to_predict_list = list(to_predict_list.values())
    data = to_predict_list
    movie=data[0]
    movie_hashtag='#'+movie.replace('/n','').replace(' ','')
    client=tweepy.Client(bearer_token=config.BEARER_TOKEN)
    query='review '+movie_hashtag+' -is:retweet lang:en -has:media'
    response=client.search_recent_tweets(query=query, max_results=10)
    lst_movie_reviews_from_twtr=[]
    for tweet in response.data:
        lst_movie_reviews_from_twtr.append(tweet.text)
    reconstructed_model = keras.models.load_model("sentiment-score-model")


    data_test = {'review':lst_movie_reviews_from_twtr}
    df_test = pd.DataFrame(data=data_test)
    df_test.head()
    df_test["review"]=df_test.review.apply(lambda x: clean_text(x))

    list_sentences_test = df_test["review"]
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
    prediction = reconstructed_model.predict(X_te)
    #print("prediction:",prediction)
    p_mean=prediction.mean()
    final_pred='Not Defined'
    if p_mean>=0.6:
        final_pred='Must Watch. '+'Average Review Score:',p_mean*100+'%'
    if p_mean>0.3 and p_mean<0.6:
        final_pred='Maybe Watch. '+'Average Review Score:',p_mean*100+'%'
    if p_mean<=0.3:
        final_pred='Not Worth It. ','Average Review Score:',p_mean*100+'%'

    return render_template('index.html',prediction_text="Below are prediction results: "+str(final_pred))


if __name__ == "__main__":
    app.run(debug=True)
