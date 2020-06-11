import pandas as pd 
import numpy as np 
import pulp
import time
import itertools

def cal_cost(df, row_idx):
    subset = df.iloc[list(row_idx),:]
    return subset.nunique().ne(1).sum() * subset.shape[0]

def get_all_combination(df, k):
    possible_tables = []
    # Get all the row indices from dataframe
    row_idx = [i for i in range(len(df))]
    # Generate all the combinations size [k, 2k) of all the row idx
    possible_tables += [tuple(c) for r in range(k,2*k) for c in itertools.combinations(row_idx, r)]
    return possible_tables

k = 2
df = pd.read_csv('adult.csv', index_col = False, header = None)

df = df.iloc[:5,:5]
all_row_idx = get_all_combination(df, k)
test_idx = all_row_idx[15]
print(test_idx)
print(df.iloc[list(test_idx)])
cost = cal_cost(df, test_idx)
print(cost)
