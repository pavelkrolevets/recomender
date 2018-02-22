import pandas as pd
import numpy as np
import random
from scipy.sparse import rand
import matplotlib.pyplot as plt
import csv
import scipy.sparse as sparse
from data_utils_MF import make_train, auc_score, calc_mean_auc
import implicit
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('./data/dataset_users_match.csv', sep=',', index_col=0)
interact = pd.read_csv('./data/inter_matr.csv', sep=',', index_col=0)
colnames = list(interact.columns.values)
# for i in range(len(colnames)):
#     colnames[i] = colnames[i]+str('_chosen')
# interact.columns = colnames

interact_sparse = sparse.csr_matrix(interact)

users_chosen_train, users_chosen_test, users_users_altered = make_train(interact_sparse, pct_test = 0.2)

alpha = 15
user_vecs, users_chosen_vecs = implicit.alternating_least_squares((users_chosen_train*alpha).astype('double'),
                                                          factors=20,
                                                          regularization = 0.1,
                                                         iterations = 200)

user_vecs = pd.DataFrame(user_vecs)
user_vecs.index = list(interact.index.values)
users_chosen_vecs = pd.DataFrame(users_chosen_vecs)
users_chosen_vecs.index = list(interact.columns.values)


"""Make recomendations"""
user_id = 141
position = list(interact.index.values).index(user_id)
num_items = 10
pref_vec = users_chosen_train[position, :].toarray()  # Get the ratings from the training set ratings matrix
pref_vec = pref_vec.reshape(-1) + 1  # Add 1 to everything, so that items not purchased yet become equal to 1
pref_vec[pref_vec > 1] = 0  # Make everything already purchased zero
rec_vector = np.dot(user_vecs.loc[user_id],users_chosen_vecs.T)  # Get dot product of user vector and all item vectors
# Scale this recommendation vector between 0 and 1
min_max = MinMaxScaler()
rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
recommend_vector = pref_vec * rec_vector_scaled
# Items already purchased have their recommendation multiplied by zero
product_idx = np.argsort(recommend_vector)[::-1][:num_items]  # Sort the indices of the items into order
rec_list = []  # start empty list to store items
user_chosen_list = list(interact.columns.values)
for index in product_idx:
    rec_list.append(user_chosen_list[index])
    # Append our descriptions to the list
print(rec_list)

"""Get chosen users for a given user"""
chosen_users_list = users_chosen_train[position, :].toarray()
chose_list = []
for i in range(len(chosen_users_list[0,:])):
    if chosen_users_list[0, i] == 1:
        user = list(interact.columns.values)[i]
        chose_list.append(dataset.loc[int(user), ['age','gender','hair_type']])

'''Get rec_users'''
recomended_list = []
for i in rec_list:
    recomended_list.append(dataset.loc[int(i), ['age','gender','hair_type']])

precision = calc_mean_auc(users_chosen_train, users_users_altered,
              [sparse.csr_matrix(user_vecs), sparse.csr_matrix(users_chosen_vecs.T)], users_chosen_test)

print('Chosen users: ',chose_list, 'Recomended users: ', recomended_list)
print(precision)

