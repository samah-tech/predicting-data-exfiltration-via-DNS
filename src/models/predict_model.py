

from cProfile import label


def predicting_model(test_data ):
    import pickle
    import pandas as pd
    from src.features import build_features
    from kafka import KafkaProducer
    import json
    from elasticsearch import Elasticsearch, helpers
    import datetime
    from src.data import make_dataset

    # Load the model
    with open("D:\\cyber\\ass_2\\CS_ASS\\assignment2-samah-tech\\models\\saved_model", 'rb') as pickle_file:
        rf = pickle.load(pickle_file)

    test_data_clean=test_data.drop(columns=['longest_word'])
    test_data_clean=test_data_clean.drop(columns=['sld']) 

    #use saved_model to predict 
    Label = rf.predict(test_data_clean)
    confidence = rf.predict_proba(test_data_clean)
    return Label ,confidence
