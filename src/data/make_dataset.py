import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from kafka import KafkaConsumer
import pandas as pd
#from src.models import train_model

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    from src.models import train_model 
    from src.features import build_features
    from src.models import predict_model
    train_model.training_model(input_filepath)
    #_____kafka
    
    consumer = KafkaConsumer('ml-raw-dns',bootstrap_servers = ['localhost:9092'],auto_offset_reset = 'earliest',enable_auto_commit = False,)
    list_data = []
    count=0
    for message in consumer:
        print(message)
        record =message.value.decode("utf-8")
        list_data.append(record)
        count+=1
        if count==100000:break
    kafka_data = pd.DataFrame(list_data,columns=['domain']) 
    test_data = build_features.pass_data(kafka_data)
    test_data.to_csv("D:\\cyber\\ass_2\\CS_ASS\\assignment2-samah-tech\\data\\interim\\Intermediate_data.csv", index=False)
    label,score = predict_model.predicting_model(test_data)
    print(type(test_data))
    print(type(label))
    ###
    #dataframe with resulting label
    test_data.insert(0,'domain',list_data)
    ll=label.tolist()
    sscor=score[:,0].tolist()
    data_frame_new = pd.DataFrame(
    {'label': ll,
     'confidence_score': sscor})
    agg_df = pd.concat([test_data, data_frame_new], axis=1)
    #
    print(agg_df.head())
    agg_df.to_csv(output_filepath, index=False)
    # logger = logging.getLogger(__name__)
    # logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main("D:\\cyber\\ass_2\\CS_ASS\\assignment2-samah-tech\\data\\raw\\training_dataset.csv","D:\\cyber\\ass_2\\CS_ASS\\assignment2-samah-tech\\data\\processed\\final_data.csv")

