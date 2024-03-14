import numpy as np
import pandas as pd
from pprint import pprint

df = pd.read_csv('flights_dataset.csv')

#pprint(df.info())
#print(".........................................")

df.drop(['hour','minute','tailnum'],axis=1, inplace=True)

#pprint(df.infer_objects())

# verificar valores nulos no dataset
#print(df.isnull().sum())

# substituir todos os valores nulos por -99
df.fillna(-99, inplace=True)

#print(".........................................")
#print(df.isnull().sum())

#frequency distribution of categories within a feature
destinations = df['dest'].unique()
#print('Unique count: ', df['dest'].value_counts().count())
#print(df['dest'].value_counts())
#print('---------------------------------------------')



'''
Function to encode all non-(int/float) features in a dataframe.
For each column, if its dtype is neither int or float, get the list of unique values,
store the relation between the label and the integer that encodes it and apply it.
Return a labelled dataframe and a dictionary label to be able to restore the original value.
(8-10 lines - you may need an auxiliar function)
'''
def label_encoding(df):
    label_dictionary = {}
    
    for column in df.columns:
        if(df[column].infer_objects().dtype == 'object'):
            unique_values = df[column].unique()
            label_dictionary[column] = {}
            for i, val in enumerate(unique_values):
                label_dictionary[column][i] = val
                df[column] = df[column].replace(val, i)
                
    return df, label_dictionary

'''
Function to decode what was previously encoded - get the original value!
'''
def label_decoding(df_labelled, label_dictionary):
    
    for column in label_dictionary:
        for key in label_dictionary[column]:
            label = label_dictionary[column][key]
            df_labelled[column] = df_labelled[column].replace(key, label)
        
    return df_labelled


df_labelled, label_dictionary = label_encoding(df)

df_labelled_decoded = label_decoding(df_labelled, label_dictionary)

print(df.columns.values)

df_pandas_ohe = pd.get_dummies(df, columns=['carrier', 'origin', 'dest'], dtype=int)
print(df_pandas_ohe.head())