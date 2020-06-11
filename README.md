# CS517_final_project

This is the course project for the CS517 : complexity theory course 

Contributors : Hoang Le, Rajesh Mangannavar

Aim : To k-anonymize a given databse by reducing it to set cover and solving it using Integer linear programming. 

Idea : We convert our k-anonymization problem into a set cover problem. We then solve the set cover problem using integer linear programming methods from cvx. We convert back the solution of set cover to k-anonymized databse 

Package requirements :

1. cvxpy==1.0.25
2. pandas
 
Running the experiment : 

python cvx_solver.py

To try different size of the dataset, change value of n,m and k in the code

n = Number of rows in the databse
m = number of columns in the database
k = degree of anonymization
