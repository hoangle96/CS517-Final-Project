# K-anonymize database through solving set cover with ILP

This is the course project for the CS517 : Theory of Complexity course 

Contributors : Hoang Le, Rajesh Mangannavar

Aim : To k-anonymize a given databse by reducing it to set cover and solving it using Integer Linear programming. 

Idea : We convert our k-anonymization problem into a set cover problem. We then solve the set cover problem using integer linear programming methods with [`cvx`](https://github.com/cvxgrp/cvxpy). We convert back the solution of set cover to a k-anonymized database.

Package requirements :
```
1. cvxpy==1.0.25
2. pandas
3. numpy
```
Running the experiment : 

```python cvx_solver.py -filename <filename.csv> -M <m> -N <n> -K <k>```

with filename being the path to the database in `.csv` format, `-M` being the number of rows, `-N` being the number of public attributes needed to be anonymized, and `-K` being the number of rows that are similar to each other. The result will be saved in `"k_anonymized_df.csv"` with k being replaced by the value of -K. We also include the *Adult* dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult).

Example Run :

''' python cvx_solver.py -filename adult.csv -M 10 -K 2 -N 5 '''

Run ''' python cvx_solver.py -- help ''' for information while running.
