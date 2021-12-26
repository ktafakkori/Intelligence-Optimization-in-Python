import gekko as op
import itertools as it

#Developer: @KeivanTafakkori, 26 December 2021

def predict(x,z,a):      
    return sum(a[i]*x[i] for i in range(len(a))) + z

def model (U,T,a,b,normalize="y",solve="y"):
    m = op.GEKKO(remote=False, name='LinearRegressionProblem') 
    x = {i: m.Var(lb=None, ub=None) for i in U}
    z = m.Var(lb=None, ub=None)
    g = {t: m.Var(lb=None,ub=None) for t in T}    
    if normalize =='y':
        ran_a = {i: max(a[t][i] for t in T) - min(a[t][i] for t in T) for i in U}
        ave_a = {i: sum(a[t][i] for t in T)/len(T) for i in U}
        n_a = {(t,i): (a[t][i]-ave_a[i])/ran_a[i] for t,i in it.product(T,U)}
        ran_b = max(b[t] for t in T) - min(b[t] for t in T)
        ave_b = sum(b[t] for t in T)/len(T)
        n_b = {t: (b[t]-ave_b)/ran_b for t in T}
    else:
        n_a = {(t,i): a[t][i] for t,i in it.product(T,U)}
        n_b = {t: b[t] for t in T}  
    objs = {0: (2*len(T))**(-1)*sum((g[t]-n_b[t])**2 for t in T)}
    cons = {0: {t: (g[t] == sum(n_a[(t,i)]*x[i] for i in U) + z) for t in T}}
    m.Minimize(objs[0])
    for keys1 in cons:
        for keys2 in cons[keys1]: m.Equation(cons[keys1][keys2])   
    if solve == "y":
        m.options.SOLVER=1
        m.solve(disp=True)
        for keys in x: 
            x[keys] =  x[keys].value[0]
            print(f"x[{keys}]", x[keys])
        z = z.value[0]
        print("z", z)
    return m,x,z

#     EXP1     EXP2    EXP3    EXP4    EXP5
a = [[1,2,2],[2,3,3],[3,4,5],[4,5,6],[5,7,8]] #Training Dataset (inputs)  
b = [ 1     , 2     , 3     , 4     , 5     ] #Training Dataset (outputs)
U = range(len(a[0]))  #Set of input features
T = range(len(b)) #Set of the training points

m,x,z = model(U,T,a,b,normalize="n") #Model and solve the problem
print(predict(x,z,[6,4,5])) #Predict the output (99.99% Accurate :)!
