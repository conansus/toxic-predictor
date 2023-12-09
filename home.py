from flask import Flask, render_template, request
import pickle
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

#from colab
import nltk
nltk.download('wordnet')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from matplotlib import style
style.use("ggplot")
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
#stop_words = set(stopwords.words("english"))
#from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#nltk.download('punkt')

data = pd.read_csv(r"C:\Users\USER\vs_workspace\nlp_flask\traintoxic.csv")

#with open("model/toxicmode.pkl","rb") as file:
#    model = pickle.load(file)

#with open("model/vectoriserfortoxic.pkl","rb") as file:
#    vectorisor = pickle.load(file)

def data_preprocessing(text):
  text =text.lower()
  text = re.sub(r"https\S+|www\S+http\S+", '', text, flags = re.MULTILINE)
  text = re.sub(r'\@w+|\#','', text)
  text = re.sub(r'[^\w\s]','',text)
  text = re.sub(r'ð','',text)
  text = word_tokenize(text)
  filtered = [word for word in text if word not in stop_words ]
  filter = " ".join(filtered)
  return filter

lemmatizer = WordNetLemmatizer()
def lemmatizing(data):   #so same thing for lemmatization, need to separate the words in the sentence, lemma using list , then join back using " ".join
    tweet = word_tokenize(data)
    tweet = [lemmatizer.lemmatize(word) for word in tweet]
    tweet = " ".join(tweet)
    return tweet

data.tweet = data.tweet.apply(data_preprocessing)  #apply on the dataset, call the function on the dataset
data = data.drop_duplicates("tweet")
data["tweet"] = data["tweet"].apply(lambda x : lemmatizing(x))  #lambda will take the text in each row of the column and enter the function
vect = TfidfVectorizer(ngram_range=(1,3)).fit(data["tweet"])
x = data["tweet"]
y = data["label"]

x=vect.transform(x)   #after vectorizer than only transform
x_train, x_test, y_train, y_test =train_test_split(x, y , train_size=0.8,test_size = 0.2, random_state=42)

import pickle
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

app = Flask(__name__, template_folder= "templates" )  #call flask class __name__ is the main class since i run thid script, file, program code
#its the main program

@app.route("/", methods =["GET","POST"]) #refers to url of the webapp, root of the url, 
#get tu method to receive data in route, post tu method to submit data to another resource 
def index(): #whatever under the app.route will be displayed
    if request.method == 'GET':
        return(render_template('index.html'))
    
    if request.method == 'POST':
        tweet = request.form['tweet']
        listtweet = [[tweet]]
        list = pd.DataFrame(listtweet, columns=["colum"])
        tweet = list.colum.apply(data_preprocessing)
        lemdata = tweet.apply(lambda x : lemmatizing(x))  #lambda will take the text in each row of the column and enter the function
        x = vect.transform(lemdata)
        pred = logreg.predict(x)
        return render_template('index.html', result=pred, original_input={'Mobile Review':tweet})

if __name__=="__main__":  #and , or && , || is not 
    app.run(debug= True)