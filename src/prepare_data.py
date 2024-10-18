import os,bentoml
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def import_raw_data(raw_data_relative_path, 
                    filenames,
                    bucket_folder_url):
    '''import filenames from bucket_folder_url in raw_data_relative_path'''
    if os.path.exists(raw_data_relative_path)==False:
        os.makedirs(raw_data_relative_path)
    # download all the files
    for filename in filenames :
        input_file = os.path.join(bucket_folder_url,filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        object_url = input_file
        print(f'downloading {input_file} as {os.path.basename(output_file)}')
        response = requests.get(object_url)
        if response.status_code == 200:
            # Process the response content as needed
            content = response.text
            text_file = open(output_file, "wb")
            text_file.write(content.encode('utf-8'))
            text_file.close()
        else:
            print(f'Error accessing the object {input_file}:', response.status_code)
        return output_file
            
def split_data(df):
    # Split data into training and testing sets
    target = df['Chance of Admit ']
    feats = df.drop(['Chance of Admit '], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if os.path.exists(output_folderpath)==False:
        os.makedirs(output_folderpath)

def save_dataframes1(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)
               
def process_data(X_train, X_test,output_filepath):
    # Import datasets
    X_train=X_train.dropna()
    X_test=X_test.dropna()
    X_train=X_train.drop_duplicates()
    X_test=X_test.drop_duplicates()
    scaler = StandardScaler()
    print(X_train.head())
    
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
    # Enregistrer le modèle dans le Model Store de BentoML
    model_ref = bentoml.sklearn.save_model("admission_scaler", scaler)
    print(f"Standarscaler enregistré sous : {model_ref}")
    
    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes to their respective output file paths
    save_dataframes2(X_train_scaled, X_test_scaled, output_filepath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def save_dataframes2(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        #file.to_csv(output_filepath, index=False)
        print(output_filepath)        
        np.savetxt(output_filepath, file, delimiter=",")
        
def main(raw_data_relative_path="./data/raw", 
        filenames = ["admission.csv"],
        bucket_folder_url= "https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv?_gl=1*16jxga6*_gcl_au*MTgyNDE0NjE5NS4xNzI1NjA4MTkx"     
        ):
    """ Upload data from AWS s3 in ./data/raw
    """
    output_filepath='./data/processed'
    
    output_file=import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
        
    df = import_dataset(output_file, sep=",",header=0,index_col=0)
    
    # Split data into training and testing sets    
    X_train, X_test, y_train, y_test = split_data(df)

    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes to their respective output file paths
    save_dataframes1(X_train, X_test, y_train, y_test, output_filepath)
    
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')
    
    process_data(X_train, X_test, output_filepath)
    

    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
