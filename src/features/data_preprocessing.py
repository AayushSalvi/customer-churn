import numpy as np
from yaml import safe_load
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import sys
from outlier_removal import OutliersRemover

TARGET = 'Churn'

def save_transformer(path,object):
    joblib.dump(value=object,
                filename=path)


def remove_outliers(dataframe:pd.DataFrame, percentiles:list, column_names:list) -> pd.DataFrame:
    df = dataframe.copy()
    
    outlier_transformer = OutliersRemover(percentile_values=percentiles,
                                          col_subset=column_names)
    
    # fit on the data
    outlier_transformer.fit(dataframe)
     
    return outlier_transformer
#* Vendor -id OHE, date columns
#* lat/long - min max scale
#* distances - Standard scale

def train_preprocessor(data:pd.DataFrame):
    
    preprocessor = ColumnTransformer(transformers=[
        ('standard-scale',StandardScaler(),data.columns)
    ],remainder='passthrough'   )
    
    # set the output as df
    preprocessor.set_output(transform='pandas')
    # fit the preprocessor on training data
    preprocessor.fit(data)
    
    return preprocessor

def transform_data(transformer,data:pd.DataFrame):
    
    # transform the data
    data_transformed = transformer.transform(data)
    
    return data_transformed


def read_dataframe(path):
    df = pd.read_csv(path)
    return df

def save_dataframe(dataframe:pd.DataFrame, save_path):
    dataframe.to_csv(save_path,index=False)

    
def main():
    # current file path
    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent
    # input_data path
    input_path = root_path / 'data' / 'interim' 
    # read from the parameters file
    with open('params.yaml') as f:
        params = safe_load(f)
    # percentile values
    percentiles = list(params['data_preprocessing']['percentiles'])
    # save transformers path
    save_transformers_path = root_path / 'models' / 'transformers'
    # make directory
    save_transformers_path.mkdir(exist_ok=True)
    # save output file path
    save_data_path = root_path / 'data' / 'processed' / 'final'
    # make directory
    save_data_path.mkdir(exist_ok=True)
    
    for filename in sys.argv[1:]:
        complete_input_path = input_path / filename
        if filename == 'train.csv':
            df = read_dataframe(complete_input_path)
            # make X and y
            X = df.drop(columns=TARGET)
            y = df[TARGET]
            # remove outliers from data
            outlier_transformer = remove_outliers(dataframe=X,percentiles=percentiles,
                                                  column_names=X.columns)
            # save the transformer
            save_transformer(path=save_transformers_path / 'outliers.joblib',
                             object=outlier_transformer)
            # transform the data
            df_without_outliers = transform_data(transformer=outlier_transformer,
                                                 data=X)                
            # train the preprocessor on the data
            preprocessor = train_preprocessor(data=df_without_outliers)
            # save the preprocessor
            save_transformer(path=save_transformers_path / 'preprocessor.joblib',
                             object=preprocessor)
            # transform the data
            X_trans = transform_data(transformer=preprocessor,
                                     data=X)
            # fit the target transformer
            # output_transformer = transform_output(y)
            # # transform the target
            # y_trans = transform_data(transformer=output_transformer,
            #                          data=y.values.reshape(-1,1))
            # save the transformed output to the df
            X_trans['Churn'] = y
            # save the output transformer
            save_transformer(path=save_transformers_path / 'output_transformer.joblib',
                             object=X_trans)
            
            # save the transformed data
            save_dataframe(dataframe=X_trans,
                           save_path=save_data_path / filename)
            
        # elif filename == 'test.csv':
        #     df = read_dataframe(complete_input_path)
        #     # make X and y
        #     X = df.drop(columns=TARGET)
        #     y = df[TARGET]
        #     # load the transfomer
        #     outlier_transformer = joblib.load(save_transformers_path / 'outliers.joblib')
        #     df_without_outliers = transform_data(transformer=outlier_transformer,
        #                                         data=X)                
        #     # load the preprocessor
        #     preprocessor = joblib.load(save_transformers_path / 'preprocessor.joblib')
        #     # transform the data
        #     X_trans = transform_data(transformer=preprocessor,
        #                             data=X)
        #     # load the output transformer
        #     output_transformer = joblib.load(save_transformers_path / 'output_transformer.joblib') 
        #     # transform the target
        #     y_trans = transform_data(transformer=output_transformer,
        #                             data=y.values.reshape(-1,1))
        #     # save the transformed output to the df
        #     X_trans['trip_duration'] = y_trans
            
        #     # save the transformed data
        #     save_dataframe(dataframe=X_trans,
        #                 save_path=save_data_path / filename)
            
        elif filename == 'test.csv':
            df = read_dataframe(complete_input_path)
            X = df.drop(columns=TARGET)
            y = df[TARGET]
            # load the transfomer
            outlier_transformer = joblib.load(save_transformers_path / 'outliers.joblib')
            df_without_outliers = transform_data(transformer=outlier_transformer,
                                                data=df)                
            # load the preprocessor
            preprocessor = joblib.load(save_transformers_path / 'preprocessor.joblib')
            # transform the data
            X_trans = transform_data(transformer=preprocessor,
                                    data=X)
            X_trans['Churn'] = y
            # save the transformed data
            save_dataframe(dataframe=X_trans,
                        save_path=save_data_path / filename)
            
if __name__ == "__main__":
    main()