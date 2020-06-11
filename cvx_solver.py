import numpy as np
import cvxpy as cp
import pandas as pd
import time
import itertools
from pathlib import Path
def nunique_percol_sort(a):
    b = np.sort(a,axis=0)
    return (b[1:] != b[:-1]).sum(axis=0)+1

def cal_cost(df, row_idx):
    # row_idx_int = [int(c) for c in row_idx]
    subset = df.iloc[row_idx,:]
    subset = subset.to_numpy(dtype=np.int)
    nunique_per_row = nunique_percol_sort(subset)
    return  ((nunique_per_row>1).sum())*subset.shape[0]
    # return subset.nunique().ne(1).sum() * subset.shape[0]

def get_all_combination(df, k):
    possible_tables = []
    # Get all the row indices from dataframe
    row_idx = [i for i in range(len(df))]
    # Generate all the combinations size [k, 2k) of all the row idx
    possible_tables += [list(c) for r in range(k,2*k) for c in itertools.combinations(row_idx, r)]
    return possible_tables

def convert_combination(df_cost_cal, df, combination):
    #print (combination)
    subset = df_cost_cal.iloc[combination,:]
    subset = subset.to_numpy(dtype=np.int)
    # nunique_per_row = nunique_percol_sort(subset)

    for j in range(0,len(df.columns)):
        flag = 0
        for i in range(len(combination)-1):
            if subset[i][j] != subset[i+1][j] :
                flag = 1
                break
        if flag == 1:
            for elem in combination :
                df.iat[elem,j] = "*"
    return
    

if __name__ == "__main__":
    n = 50
    m = 10
    k = 3

    # DATA
    # df = pd.read_csv(Path('/adult.data'), index_col = False, header = None)
    df = pd.read_csv(Path('adult.data'), index_col = False, header = None)
    # print(len(df.columns))
    df = df.dropna()
    df = df.iloc[:n,:m]
    print(df)
    df_to_cal_cost = df.apply(lambda x: pd.factorize(x)[0])
    # print(df_to_cal_cost)

    algo_1_st = time.time()
    possible_combinations = get_all_combination(df, k)
    n_rows = len(df)
    n_combo = len(possible_combinations)

    weights = np.array([cal_cost(df_to_cal_cost, idx) for idx in possible_combinations]).reshape((1,-1))
    row_combo_map = np.zeros((n_rows, n_combo))

    # create the row_combo map
    for r in np.arange(n_rows):
        for idx, c in enumerate(possible_combinations):
            if r in c:
                row_combo_map[r, idx] = 1

    algo_1_endtime = time.time()
    print("algo 1",algo_1_endtime - algo_1_st) 

    # VARS
    x = cp.Variable((n_combo, 1), boolean = True)
    # CONSTRAINS
    cons = [cp.sum(row_combo_map@x, axis=1) == 1]
    # Define problem
    problem = cp.Problem(cp.Minimize(weights@x), cons)

    start_t = time.time()
    problem.solve()
    end_t = time.time()
    print("solver time", end_t - start_t)
    # print(problem.status)
    if problem.status == 'optimal':
    # print(x.value)
        solution = []

        sol_st = time.time()
        df = df.astype(str)
        for idx, combo in enumerate(possible_combinations):
            if np.isclose(x.value[idx], 1):
                solution.append(combo)
                convert_combination(df_to_cal_cost,df, combo)
        sol_end = time.time()
        print("algo 3 run time", sol_end - sol_st)
        print(df)
        file_name = str(k)+"_anonymized_df.csv"
        df.to_csv(file_name, index = False)
    else:
        print("No optimal solution")


    # each row_idx is in only one combination
