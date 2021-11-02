import pickle
from flask import Flask
from flask import request
from flask import jsonify

import pandas as pd

model_file = 'model.bin'
dict_file = 'dv.bin'
test_samples = 'test_samples.bin'

with open(model_file, 'rb') as infile:
    model = pickle.load(infile)

with open(dict_file, 'rb') as infile:
    dv = pickle.load(infile)

# with open(test_samples, 'rb') as infile:
#     samples = pickle.load(infile)

scaled_cols = ['item_cat_0',\
   'item_cat_1', 'item_cat_2', 'item_cat_3',  'item_w_scale', 'item_type', 'item_mrp_scale', 'outlet_id',\
   'out_size', 'out_type','outlet_age_scale']

app = Flask('store_sales')

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
    for col in scale_cols:
        df[col+'_scale'] = (df[col] - min(df[col])) / (max(df[col]) - min(df[col]))
    return df

def process_data(df, cols):
    """

    """
    outlet_size_map = {'Small': 1, 'Medium': 2, 'High': 3}
    outlet_location_map = {'Tier 1': 3,'Tier 2': 2, 'Tier 3': 1}

    scale_cols = ['item_w', 'item_mrp', 'outlet_age']



    df = pd.DataFrame.from_dict(df).reset_index(drop= True)
    df_ids =  pd.DataFrame(retrieve_item_cats(df, 'item_id')).reset_index(drop= True)
    df_id_split = pd.concat([df, df_ids], axis= 1)

    df_id_split['out_size'] = df['outlet_size'].map(outlet_size_map)
    df_id_split['out_type'] = df['outlet_location_type'].map(outlet_location_map)
    df_id_split['outlet_age']= 2021-df_id_split['outlet_year']

    df_id_split = scale_num_cols(df_id_split, scale_cols)

    return df_id_split[scaled_cols]


@app.route('/predict', methods= ['POST'])
def predict():
    item_details = request.get_json()


    items = process_data(item_details, scaled_cols)
    items_dict = items.to_dict(orient= 'records')


    X = dv.transform(items_dict)

    y_pred = model.predict(X)

    y_pred = [round(val, 2) for val in y_pred.tolist()]

    item_ids = [sample['item_id'] for sample in item_details]

    result = {
        'sales': dict(zip(item_ids, y_pred))
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug= True, host= '0.0.0.0', port= 5501)
