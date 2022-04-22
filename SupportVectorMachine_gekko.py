import gekko as op
import itertools as it

#Developer: @KeivanTafakkori, 22 April 2022

def model (U,T,a,b,solve="y"):
    m = op.GEKKO(remote=False, name='SupportVectorMachine') 
    alpha = {t: m.Var(lb=0, ub=None) for t in T}
    n_a = {(t,i): a[t][i] for t,i in it.product(T,U)}
    n_b = {t: b[t] for t in T}  
    objs = {0: sum(alpha[t] for t in T) - sum(alpha[t]*alpha[tt] * n_b[t]*n_b[tt] * sum(n_a[(t,i)]*n_a[(tt,i)] for i in U) for t,tt in it.product(T,T))}
    cons = {0: {0: ( sum(alpha[t]*n_b[t] for t in T) == 0) for t in T}}
    m.Maximize(objs[0])
    for keys1 in cons:
        for keys2 in cons[keys1]: m.Equation(cons[keys1][keys2])   
    if solve == "y":
        m.options.SOLVER=1 
        m.solve(disp=True)
        for keys in alpha: 
            alpha[keys] =  alpha[keys].value[0]
            print(f"alpha[{keys}]", alpha[keys])
    x = [None for i in U]
    for i in U:
        x[i]=sum(alpha[t]*b[t]*n_a[(t,i)] for t in T)
    for t in T:
        if alpha[t]>0:
            z=b[t] - sum(x[i]*n_a[(t,i)] for i in U)
            break
    return m,x,z,alpha

def classify(dataset,x,z,alpha,a): 
   if sum(sum(alpha[t]*dataset[1][t]*dataset[0][t][i] for t in T)*a[i] for i in U) + z > 0:   
    return 1
   else:
    return -1

#     EXP1     EXP2    EXP3    EXP4    EXP5
a = [[1,2,2],[2,3,3],[3,4,5],[4,5,6],[5,7,8]] #Training Dataset (inputs)  
b = [ -1     , 1     , -1     , 1     , -1     ] #Training Dataset (outputs)
U = range(len(a[0]))  #Set of input features
T = range(len(b)) #Set of the training points

m, x, z, alpha = model(U,T,a,b) #Model and solve the problem

print(classify([a,b],x,z,alpha,[1,2,2])) #Predict the output (100% Accurate :)!
