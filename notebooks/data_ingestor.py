import os
import pandas as pd
import json


def read_all_json_files(path, column_dict, inconsistent_col, keys):
    '''
    Reads all the json files give directory.
    Returns a dataframe with all the data combined
    '''
    list_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    sorted_list_files = sorted(list_files)
    df_total = pd.DataFrame(columns = keys)
    for file in sorted_list_files:
        with open(path+file, 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
        for i in inconsistent_col:
            if i in df.columns:
                df = df.rename(columns=column_dict)
        df_total = df_total.append(df, ignore_index=True)
    print(df_total.shape)
    return df_total

def maintain_datatype(df):
    df = df.astype({ "country": str, "day": int,
                            "month": int, "price": float,
                            "times_viewed": int, "year": int})
    return df

def drop_non_numeric_invoice(df):
    df = df[~df['invoice'].str.contains("[a-zA-Z]").fillna(False)]
    return df
