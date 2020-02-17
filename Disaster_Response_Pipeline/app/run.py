import json
import plotly
import pandas as pd
import numpy as np 
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download(['punkt', 'stopwords'])
nltk.download('averaged_perceptron_tagger')
stop_words = stopwords.words('english')
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, make_scorer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.graph_objs as graph_o
from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import Counter
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import sys

sys.path.append("models")

from Start_Verb_Extractor import StartVerbExtractor
from text_len_get import get_text_len

app = Flask(__name__)

def tokenize(text):
    """
    Normalize, tokenize and stems texts.
    
    Input:
    text: string. Sentence containing a message.
    
    Output:
    stemmed_tokens: list of strings. A list of strings containing normalized and stemmed tokens.
    """
    
    # Normalizing the text.
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return stemmed_tokens

def get_counted_words():
    counted_words = np.load('data/counts.npz')
    return list(counted_words['top_words']), list(counted_words['top_counts'])

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('message_categories', engine)

# load model
model = joblib.load("models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
   
    # Calculate proportion of each category with label = 1
    cat_prop = df.drop(['message'], axis = 1).sum()/len(df)
    cat_prop = cat_prop.sort_values(ascending = False)
    cat_names = list(cat_prop.index)
    
    # top 20 words and counts 
    top_words, top_counts = get_counted_words()
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_prop
                )
            ],

            'layout': {
                'title': 'Proportion of Messages by Category',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words,
                    y=top_counts
                )
            ],

            'layout': {
                'title': 'Top 20 words by count',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Words",
                    'tickangle': -45
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')  

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[1:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()