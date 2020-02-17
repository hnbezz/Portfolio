import sys
import pandas as pd
import numpy as np
import re
import pickle
import warnings

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

from sqlalchemy import create_engine
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

nltk.download(['punkt', 'stopwords'])
nltk.download('averaged_perceptron_tagger')
stop_words = stopwords.words('english')
warnings.simplefilter('ignore')

def load_data(database_filepath):
    """
    Load the data
    
    Inputs:
    database_filepath: String. Filepath for the db file containing the cleaned data.
    
    Output:
    X: dataframe. Contains the feature data.
    y: dataframe. Contains the labels (categories) data.
    category_names: List of strings. Contains the labels names.
    """
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message_categories', engine)
    
    X = df['message']
    y = df.drop(['message','offer','request'], axis=1)
    category_names = y.columns.tolist()
    
    return X, y, category_names

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
    
def get_text_len(data):
    """ 
    Gets the text length.
    """
    
    return np.array([len(text) for text in data]).reshape(-1, 1)

def recall_metric(y_true, y_pred):
    """
    Calculate mean Recall 
    
    Args:
    y_true: array. Array containing actual labels.
    y_pred: array. Array containing predicted labels.
        
    Returns:
    score: float. Median F1 score for all of the output classifiers
    """
    score_list = []
    for i in range(np.shape(y_pred)[1]):
        score_i = (recall_score(np.array(y_true)[:, i], y_pred[:, i]) * 3 + 
                   precision_score(np.array(y_true)[:, i], y_pred[:, i])) / 4
        score_list.append(score_i)
        
    score = np.mean(score_list)
    return score

def build_model():
    """
    Builds a ML pipeline, makes a scorer and performs a gridsearch for the best parameters.
    
    Input:
    None
    
    Output:
    cv: gridsearchcv object. Object that transforms and fit the data, creates the model and finds the best parameters among
    the ones searched.
    """
    
    # Creates pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('start_verb', StartVerbExtractor()),
            ('length', Pipeline([('text_length', FunctionTransformer(get_text_len, validate=False))]))
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # Parameters
    parameters = {'features__text_pipeline__vect__ngram_range':[(1,2),(2,2)], 'clf__estimator__n_estimators':[50, 100, 300]}

    # Creates scorer
    scorer = make_scorer(recall_metric)
    
    np.random.seed(42)
    
    # Creates gridsearch
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, n_jobs=8, verbose=10)
    
    return cv
    
    
def metricss(test_labels, predicted_labels, col_names):
    """
    Creates a dataframe with metrics of the model
    
    Input:
    test_labels: pandas dataframe with the test (true) labels
    predicted_labels: pandas dataframe with the predicted labels by the model
    col_names: list of strings with the names for each of the label classes.
    
    Output:
    metrics_df: dataframe containing the metrics.
    """
    
    metrics = []
    for i in range(len(col_names)):
        sum1s = sum(test_labels.iloc[:, i])
        sum0s = len(test_labels) - sum1s
        accuracy = accuracy_score(test_labels.iloc[:, i], predicted_labels.iloc[:, i])
        precision = precision_score(test_labels.iloc[:, i], predicted_labels.iloc[:, i])
        recall = recall_score(test_labels.iloc[:, i], predicted_labels.iloc[:, i])
        f1 = f1_score(test_labels.iloc[:, i], predicted_labels.iloc[:, i])
        
        metrics.append([sum1s, sum0s, accuracy, precision, recall, f1])
    
    col_names.append('mean')
    
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, columns = ['# 1s', '# 0s', 'Accuracy', 'Precision', 'Recall', 'F1'])
    metrics_df = metrics_df.append({'# 1s': sum(metrics[:,0]/len(metrics[:,0])), 
                                    '# 0s': sum(metrics[:,1]/len(metrics[:,1])), 
                                    'Accuracy': sum(metrics[:,2]/len(metrics[:,2])),
                                    'Precision': sum(metrics[:,3]/len(metrics[:,3])),
                                    'Recall': sum(metrics[:,4]/len(metrics[:,4])),
                                    'F1': sum(metrics[:,5]/len(metrics[:,5]))}, ignore_index=True)
    
    metrics_df.index = col_names
    
    return metrics_df

def evaluate_model(model, X_test, y_test, category_names):
    """
    Returns test accuracy, number of 1s and 0s, recall, precision and F1 Score.
    
    Inputs:
    model: model object. Instanciated model.
    X_test: pandas dataframe containing test features.
    y_test: pandas dataframe containing test labels.
    category_names: list of strings containing category names.
    
    Returns:
    None
    """
    
    y_pred = model.predict(X_test)
    y_pred_pd = pd.DataFrame(y_pred, columns=category_names)
    
    
    metricss(y_test, y_pred_pd, category_names)


def save_model(model, model_filepath):
    
    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

        for train_index, test_index in msss.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y.values[train_index], Y.values[test_index]
        Y_train = pd.DataFrame(Y_train,columns=category_names)
        Y_test = pd.DataFrame(Y_test,columns=category_names)
                
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        print(evaluate_model(model, X_test, Y_test, category_names))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()