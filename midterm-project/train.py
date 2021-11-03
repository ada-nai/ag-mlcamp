# import packages
import os

import math
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestRegressor

import pickle

import logging
logging.basicConfig(filename='./logs/train.log', level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def std_col_names(df):
    """
    Convert column names to lower case for the sake of convenience
    """
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    return df



def retrieve_item_cats(df, col):
    """
    Split the item_id feature into categories
    """
    item_cats = {}
    for x in range(4):
        if x<3:
            item_cats[f'item_cat_{x}'] = [df[col][loc][x] for loc in range(len(df))]
        else:
            item_cats[f'item_cat_{x}'] = [df[col][loc][x:] for loc in range(len(df))]
    return item_cats

def scale_num_cols(df, scale_cols):
    """
    Use min-max normalization
    """
    for col in scale_cols:
        df[col+'_scale'] = (df[col] - min(df[col])) / (max(df[col]) - min(df[col]))
    return df

def process_data(df, cols):
    """
    Transformations done:
        Split item_id -> retrieve_item_cats
        Ordinally encode -> pd.map
        Outlet age -> 2021 - x['outlet_year']
        One hot encode/DictVectorize columns
    """
    outlet_size_map = {'Small': 1, 'Medium': 2, 'High': 3}
    outlet_location_map = {'Tier 1': 3,'Tier 2': 2, 'Tier 3': 1}

    # columns to be scaled
    scale_cols = ['item_w', 'item_mrp', 'outlet_age']

    df_ids =  pd.DataFrame(retrieve_item_cats(df, 'item_id')).reset_index(drop= True)
    df_id_split = pd.concat([df, df_ids], axis= 1)

    df_id_split['out_size'] = df['outlet_size'].map(outlet_size_map)
    df_id_split['out_type'] = df['outlet_location_type'].map(outlet_location_map)
    df_id_split['outlet_age']= 2021-df_id_split['outlet_year']

    df_id_split = scale_num_cols(df_id_split, scale_cols)

    return df_id_split[cols]

def main():
    """
    Driver utility for training model
    """
    # feature list after all processing is performed
    # the model will be trained on these feature columns
    scaled_cols = ['item_cat_0',\
       'item_cat_1', 'item_cat_2', 'item_cat_3',  'item_w_scale', 'item_type', 'item_mrp_scale', 'outlet_id',\
       'out_size', 'out_type','outlet_age_scale']

    df = std_col_names(pd.read_csv('train.csv'))
    logging.info('Dataset loaded')

    train = process_data(df, scaled_cols)
    logging.info(f'Sample: ')
    logging.info(train.columns)

    train_dict = train.to_dict(orient= 'records')
    logging.info('Data processed into dictionary')

    # instantiate DictVectorizer
    dv = DictVectorizer(sparse= False)
    train_dv = dv.fit_transform(train_dict)
    logging.info('Data transformed using DictVectorizer')

    # instantiate model
    rf_tuned = RandomForestRegressor(ccp_alpha=0.05, max_depth=15, max_features=5,
                      max_samples=0.5, n_estimators=200)
    logging.info('Training Model')

    # fit model to transformed train data
    rf_tuned.fit(train_dv, df['sales'])
    logging.info('Model trained!')

    # export data objects required for making predictions
    model_file = 'model.bin'
    dv_file = 'dv.bin'

    with open(model_file, 'wb') as outfile:
        pickle.dump(rf_tuned, outfile)

    with open(dv_file, 'wb') as outfile:
        pickle.dump(dv, outfile)

    logging.info(f'Objects exported!: {model_file}, {dv_file}')
    print('Training completed. Check logs at ./logs/train.log')


if __name__ == "__main__":
    main()
