import numpy as np

def get_text_len(data):
    """ 
    Gets the text length.
    """
    
    return np.array([len(text) for text in data]).reshape(-1, 1)