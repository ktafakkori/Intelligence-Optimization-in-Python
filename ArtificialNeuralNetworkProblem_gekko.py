import gekko as op
import itertools as it

#Developer: @KeivanTafakkori, 23 December 2021

def model (L,U,T,a,b,lambd,solve="y"):
    m = op.GEKKO(remote=False, name='ArtificalNeuralNetworkProblem') 
    x = {l: {(i,j): m.Var(lb=None,ub=None) for i,j in it.product(U[l-1],U[l])} for l in range(1,len(L))}
    z = {l: {j: m.Var(lb=None,ub=None) for j in U[l]} for l in range(1,len(L))}
    g = {l: {(j,t): m.Var(lb=None,ub=None) for j,t in it.product(U[l],T)} for l in range(1,len(L))}  
    objs = {0: sum((g[len(L)-1][(j,t)]-b[t][j])**2 for j,t in it.product(U[len(L)-1],T)) + lambd * sum(sum(x[l][(i,j)]**2 for i,j in it.product(U[l-1],U[l])) for l in range(1,len(L)))}
    cons = {0: {0: {(j,t): g[1][(j,t)] == sum(a[t][i]*x[1][(i,j)] for i in U[0]) + z[1][j] for j,t in it.product(U[1], T)}},
            1: {l: {(j,t): g[l][(j,t)] == sum(g[l-1][(i,t)]*x[l][(i,j)] for i in U[l-1]) + z[l][j] for j,t in it.product(U[l], T)} for l in range(2,len(L))}}
    m.Minimize(objs[0])
    for keys1 in cons:
        for keys2 in cons[keys1]: 
            for keys3 in cons[keys1][keys2]: m.Equation(cons[keys1][keys2][keys3])   
    if solve == "y":
        m.options.SOLVER=1
        m.solve(disp=True)
        for keys1 in x:
            for keys2 in x[keys1]: print(f"x[{keys1}][{keys2}]", x[keys1][keys2].value)
        for keys1 in z:
            for keys2 in z[keys1]: print(f"z[{keys1}][{keys2}]", z[keys1][keys2].value)
    return m

a = [[1], [2], [3], [4], [5]]  #Training Dataset (inputs)
b = [[1], [2], [3], [4], [5]]  #Training Dataset (outputs)

L = range(2) #Set of the layers (input (features) + hidden + output)
U = [range(len(a[0])),range(1)]  #Set of neurons in the input (features), hidden, and output layers
T = range(len(b)) #Set of the training points
lambd = 0 #To prevent overfitting

m = model(L,U,T,a,b,lambd) #Model and solve the problem
