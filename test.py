"""
A set partitioning model of a wedding seating problem

Authors: Stuart Mitchell 2009
"""

import pulp
import time
import itertools
import pandas as pd
import numpy as np


# guests = 'A B C D E F G I J K L M N O P Q R Z'.split()
# guests = 'A B C D E F G I J K'.split()

def happiness(table):
    """
    Find the happiness of the table
    - by calculating the maximum distance between the letters
    """
    return abs(ord(table[0]) - ord(table[-1]))

def cal_cost(df, row_idx):
    row_idx_int = [int(c) for c in row_idx]
    subset = df.iloc[list(row_idx_int),:]
    return subset.nunique().ne(1).sum() * subset.shape[0]

def get_all_combination(df, k):
    possible_tables = []
    # Get all the row indices from dataframe
    row_idx = [str(i) for i in range(len(df))]
    # Generate all the combinations size [k, 2k) of all the row idx
    possible_tables += [tuple(c) for r in range(k,2*k) for c in itertools.combinations(row_idx, r)]
    return possible_tables

if __name__ == "__main__":
    # read file, get all possible combinations
    k = 2
    df = pd.read_csv('adult.csv', index_col = False, header = None)

    df = df.iloc[:5,:5]
    # print(df)
    all_row_idx = [str(i) for i in range(len(df))]
    possible_combinations = get_all_combination(df, k)
    # print(possible_combinations)
    #create list of all possible tables
    # create_start_time = time.time()
    # time_1_start = time.time()
    # possible_tables = [tuple(c) for c in pulp.allcombinations(guests, 
    #                                         max_table_size) if len(c) >= k]
    # print(possible_tables)
    # time_1_end = time.time()
    # print('time 1', time_1_end-time_1_start)


    #create a binary variable to state that a table setting is used
    time_2_start = time.time()
    x = pulp.LpVariable.dicts('table', possible_combinations, 
                                lowBound = 0,
                                upBound = 1,
                                cat = pulp.LpInteger)
    time_2_end = time.time()
    print('Adding vars', time_2_end-time_2_start)

    time_3_start = time.time()
    seating_model = pulp.LpProblem("Wedding_Seating_Model", pulp.LpMinimize)
    time_3_end = time.time()
    print('Construct Pulp problem', time_3_end-time_3_start)

    # time_4_start = time.time()
    # # objective = 
    # time_4_end = time.time()
    # print('time 4', time_4_end-time_4_start)

    time_5_st = time.time()
    objective = [cal_cost(df, idx) * x[idx] for idx in possible_combinations]
    time_5_end = time.time()
    print('making objective', time_5_end-time_5_st)

    time_6_start = time.time()
    seating_model += sum(objective)
    time_6_end = time.time()
    print('adding objective', time_6_end-time_6_start)

    # print(seating_model)

    # specify the maximum number of tables
    # seating_model += sum([x[table] for table in possible_tables]) #<= max_tables, \ "Maximum_number_of_tables"

    # A guest must seated at one and only one table
    guest_start_time = time.time()
    for row in all_row_idx:
        seating_model += sum([x[row_combination] for row_combination in possible_combinations if row in row_combination]) == 1
    create_guest_end_time = time.time()
    print("Add constrain of one element in only one set", create_guest_end_time-guest_start_time)


    start_time = time.time()
    seating_model.solve()
    end_time = time.time()
    print("solving time", end_time-start_time)

    print (("Status:"), pulp.LpStatus[seating_model.status])
    print("The choosen tables are out of a total of "+str(len(possible_combinations)))
    for table in possible_combinations:
        # print(x[table].value())
        if x[table].value() == 1.0:
            print(table)


