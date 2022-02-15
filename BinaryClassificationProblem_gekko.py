import gekko as op
import itertools as it
import math as mt

#Developer: @KeivanTafakkori, 15 February 2022

def classify(x,z,a):      
    return round((1+mt.exp(-(sum(a[i]*x[i] for i in U) + z)))**(-1))

def model (U,T,a,b,lam,normalize="y",regularize="y",solve="y"):
    m = op.GEKKO(remote=False, name='BinaryClassificationProblem') 
    x = {i: m.Var(lb=None, ub=None) for i in U}
    z = m.Var(lb=None, ub=None)
    g = {t: m.Var(lb=None,ub=None) for t in T}    
    n_a = {(t,i): a[t][i] for t,i in it.product(T,U)}
    n_b = {t: b[t] for t in T}  
    if regularize == "n":
        objs = {0: sum((g[t]-n_b[t])**2 for t in T)}
        #objs = {0: sum(n_b[t]*m.log(g[t])+(1-b[t])*m.log(1-g[t]) for t in T)}
    else:
        objs = {0: sum((g[t]-n_b[t])**2 for t in T) + lam*sum(x[i]**2 for i in U)}
        #objs = {0: sum(n_b[t]*m.log(g[t])+(1-b[t])*m.log(1-g[t]) for t in T) + lam*sum(x[i]**2 for i in U)}
    cons = {0: {t: (g[t] == (1+m.exp(-(sum(n_a[(t,i)]*x[i] for i in U) + z)))**(-1)) for t in T}}
    m.Minimize(objs[0])
    for keys1 in cons:
        for keys2 in cons[keys1]: m.Equation(cons[keys1][keys2])   
    if solve == "y":
        m.options.SOLVER=2
        m.solve(disp=True)
        for keys in x: 
            x[keys] =  x[keys].value[0]
            print(f"x[{keys}]", x[keys])
        z = z.value[0]
        print("z", z)
    return m,x,z

#     EXP1     EXP2    EXP3    EXP4    EXP5
a = [[1,2,2],[2,3,3],[3,4,5],[4,5,6],[5,7,8]] #Training Dataset (inputs)  
b = [ 0     , 1     , 0     , 1     , 0     ] #Training Dataset (outputs)
U = range(len(a[0]))  #Set of input features
T = range(len(b)) #Set of the training points
lam = 0 #For regularization (to avoid under/overfitting)

m,x,z = model(U,T,a,b,lam) #Model and solve the problem
print(classify(x,z,[1,2,2])) #Predict the output (100% Accurate :)!
