from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd 
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download(['punkt', 'stopwords'])
nltk.download('averaged_perceptron_tagger')
stop_words = stopwords.words('english')

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

class StartVerbExtractor(BaseEstimator, TransformerMixin):


    def start_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) != 0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
        return 0


    def fit(self, X, y=None):
        return self
    

    def transform(self, X):
        X_tag = pd.Series(X).apply(self.start_verb)
        return pd.DataFrame(X_tag)