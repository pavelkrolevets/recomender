import pandas as pd
import numpy as np
import random
from scipy.sparse import rand
import matplotlib.pyplot as plt
import csv

columns = ['user_id', 'FB_id', 'token', 'device_token', 'gender', 'email', 'f_name', 'l_name',
           'age', 'hair_type', 'eye_color', 'height', 'sexual_orient', 'nationality', 'description',
           'lookalike_score', 'date_score', 'num_likes', 'num_scores_recieved', 'num_scores_given',
           'time_last_active','location', 'is_trans', 'face_group', 'distance']

dataset = pd.DataFrame(np.zeros((100, len(columns))), columns=columns)

dataset['user_id'] = np.random.choice(range(1000), dataset.shape[0], replace=False)
dataset['FB_id'] = np.random.choice(range(1000000), dataset.shape[0], replace=False)
dataset['token'] = np.random.choice(range(100000), dataset.shape[0], replace=False)
dataset['device_token'] = np.random.choice(range(1000000), dataset.shape[0], replace=False)

gender = ['M', 'F']
for i in range(len(dataset["gender"])):
    dataset.loc[i,"gender"]= random.choice(gender)


with open('./data/CSV_Database_of_First_Names.csv', 'r') as f:
    reader = csv.reader(f)
    f_names = list(reader)

for i in range(len(dataset["f_name"])):
    dataset.loc[i,"f_name"]= random.choice(f_names)


with open('./data/CSV_Database_of_Last_Names.csv', 'r') as f:
    reader = csv.reader(f)
    l_name = list(reader)

for i in range(len(dataset["l_name"])):
    dataset.loc[i,"l_name"]= random.choice(l_name)

dataset['age'] = np.random.choice(range(18,40), dataset.shape[0], replace=True)

hair_type = ['Black hair', 'Natural black hair', 'Deepest brunette hair', 'Dark brown hair', 'Medium brown hair',
             'Lightest brown hair', 'Natural brown hair', 'Light brown hair', 'Chestnut brown hair', 'Light chestnut brown hair',
             'Auburn hair', 'Copper hair', 'Red hair', 'Titian hair', 'Strawberry blond hair', 'Light blonde hair', 'Dark blond hair',
             'Golden blond hair', 'Medium blond hair', 'Grey hair', 'White hair']
for i in range(len(dataset["hair_type"])):
    dataset.loc[i,"hair_type"]= random.choice(hair_type)


eye_color = ['Amber', 'Blue', 'Brown', 'Gray', 'Green', 'Hazel', 'Red and violet']
for i in range(len(dataset["eye_color"])):
    dataset.loc[i,"eye_color"]= random.choice(eye_color)

dataset['height'] = np.random.choice(range(150,200), dataset.shape[0], replace=True)

sexual_orient = ['Homo-sexual', 'Straight', 'Bi-sexual']
for i in range(len(dataset["sexual_orient"])):
    dataset.loc[i,"sexual_orient"]= random.choice(sexual_orient)

## nationality

with open('./data/natioanal.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    nationality = list(reader)
print(len(nationality[0]))
for i in range(len(dataset["nationality"])):
    dataset.loc[i,"nationality"] = random.choice(nationality[0])


####  lookalike_score
dataset['lookalike_score'] = np.random.choice(range(0,10), dataset.shape[0], replace=True)

### date_score
dataset['date_score'] = np.random.choice(range(0,10), dataset.shape[0], replace=True)

#### num_likes
dataset['num_likes'] = np.random.choice(range(0, dataset.shape[0]), dataset.shape[0], replace=True)

#### num_scores_recieved
dataset['num_scores_recieved'] = np.random.choice(range(0, dataset.shape[0]), dataset.shape[0], replace=True)

#### num_scores_given
dataset['num_scores_given'] = np.random.choice(range(0, dataset.shape[0]), dataset.shape[0], replace=True)

#### location
with open('./data/countries.txt', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    location = list(reader)
for i in range(len(dataset["location"])):
    dataset.loc[i,"location"] = random.choice(location[0])
#### face_group
dataset['face_group'] = np.random.choice(range(0, 50), dataset.shape[0], replace=True)
#### distance
dataset['distance'] = np.random.choice(range(0, 30), dataset.shape[0], replace=True)

dataset = dataset.set_index(['user_id'], drop=False)
'''saving user dataset'''
dataset.to_csv(path_or_buf='./data/dataset_users_match.csv', header=1, index=1)


"""Creating interaction matrix"""
sparse_matr  = rand(len(dataset['user_id']), len(dataset['user_id']), density=0.1, format='array')
inter_matr = pd.DataFrame(sparse_matr, index=dataset['user_id'], columns=dataset['user_id'])
inter_matr[inter_matr > 0] = 1
np.fill_diagonal(inter_matr.values, 0)
inter_matr.to_csv(path_or_buf='./data/inter_matr.csv', header=1, index=1)



