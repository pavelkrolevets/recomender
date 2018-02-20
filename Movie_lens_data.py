import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#column headers for the dataset
data_cols = ['user_id','movie_id','rating','timestamp']
item_cols = ['movie_id','movie title','release date',
'video release date','IMDb URL','unknown','Action',
'Adventure','Animation','Childrens','Comedy','Crime',
'Documentary','Drama','Fantasy','Film-Noir','Horror',
'Musical','Mystery','Romance ','Sci-Fi','Thriller',
'War' ,'Western']
user_cols = ['user_id','age','gender','occupation',
'zip code']

#importing the data files onto dataframes
users = pd.read_csv('./data/movie_lens/ml-100k/u.user', sep='|',
names=user_cols, encoding='latin-1')
item = pd.read_csv('./data/movie_lens/ml-100k/u.item', sep='|',
names=item_cols, encoding='latin-1')
data = pd.read_csv('./data/movie_lens/ml-100k/u.data', sep='\t',
names=data_cols, encoding='latin-1')
print ('Number strings = ' + str(len(data['user_id'])))
n_users = pd.unique(data['user_id'])
n_items = pd.unique(data['movie_id'])
print ('Number of users = ' + str(len(n_users)) + ' | Number of movies = ' + str(len(n_items)))
occup = pd.unique(users['occupation'])
print(occup)
data[['rating']] = np.where(data[['rating']]<5,0,1)

#Create one data frame from the three
dataset = pd.merge(pd.merge(item, data),users)
dataset = dataset.drop(columns=['video release date', 'IMDb URL', 'unknown'])
dataset ['category_hash'] = ''
for row in range(len(dataset['user_id'])):
    names = ['Action', 'Adventure','Animation','Childrens',
             'Comedy','Crime','Documentary','Drama','Fantasy',
             'Film-Noir','Horror', 'Musical','Mystery','Romance',
             'Sci-Fi','Thriller','War','Western']
    text = []
    for i in range(3,21):
        if dataset.ix[row,i] == 1:
            text.append(names[i-3])
    #print(text)
    dataset.set_value(row, 'category_hash', text)

dataset.to_csv(path_or_buf='./data/movie_lens/dataset_aligned.csv', header=0, index=0)
print(dataset.head())
