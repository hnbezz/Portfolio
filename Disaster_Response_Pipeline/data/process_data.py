import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories, then merge the files.
    
    Inputs:
    messages_filepath: String. Filepath for the csv file containing the messages.
    categories_filepath: String. Filepath for the csv file containing the categories.
    
    Output:
    df: pandas dataframe. Dataframe containing messages and respective categories.
    """
    
    messages = pd.read_csv(messages_filepath)
    categories1 = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories1, how='inner', on='id')
    
    return df

def clean_data(df):
    """
    Clean the dataframe, removing unnecessary columns, wrong and duplicate values, converting strings to numbers, among others.
    
    Inputs:
    df: pandas dataframe. Dataframe containing messages and respective categories.
    
    Output:
    df: pandas dataframe. Dataframe containing cleaned version of messages and respective categories.
    """
    
    categories = df['categories'].str.split(';', expand = True)
    
    # Getting categories names
    row = categories.iloc[0]
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Drop child_alone, since it contains only 0s.
    categories.drop('child_alone', axis = 1, inplace = True)
    
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)
    
    # Concat df and categories
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # the columns below are not necessary for the project.
    df = df.drop(['id','original','genre'],axis=1)
    
    # Very few nans (when compared to the total) So I drop them.
    df = df.dropna().reset_index()
    
    # drop column with no instances
    col_names = df.columns.tolist()[2:]
    for col in col_names:
        if df[col].any() == 0:
            df = df.drop(col, axis = 1) 
            col_names = df.columns.tolist()[2:]
    df = df.drop('index', axis = 1)
        
    #there are few occasions where related is 2. So I drop them.
    df = df[df['related'] != 2]
    
    # make labels into integers
    df.iloc[:,2:] = df.iloc[:,2:].astype('int64')
    
    return df
    
def save_data(df, database_filename):
    """
    Save the cleaned data.
    
    Input:
    df: pandas dataframe. Dataframe containing cleaned version of messages and respective categories.
    database_filename: String. Filename for the output database.
    
    Output:
    None.
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message_categories', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()