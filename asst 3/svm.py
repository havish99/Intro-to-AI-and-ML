
# coding: utf-8

# In[2]:


import cvxpy as cp
import numpy as np
import csv

def read_data(filename):
    X=[]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            X.append(np.array(row).astype('float64'))
    X=np.array(X)
    return X

X=read_data('Xsvm.csv')
y=read_data('ysvm.csv')
print(X.shape)
print(y.shape)


# In[3]:


# initialisations for convex optimization
alpha = cp.Variable(len(y))
term2 = cp.matmul(X.T,cp.matmul(cp.diag(alpha),y))
term2 = cp.norm(term2)**2


# In[4]:


term1 = cp.sum(alpha)
full = term1 - 0.5*term2


# In[5]:


constraint = cp.matmul(alpha.T,y)
Constraint = [0<=alpha,constraint == 0]
obj = cp.Maximize(full)
prob = cp.Problem(obj, Constraint)
prob.solve(verbose=True)


# In[8]:


Alpha=np.array(alpha.value).reshape(500,)


# In[14]:


w=np.dot(X.T,np.matmul(np.diag(Alpha),y))
w=w.reshape(2,1)


# In[15]:


Alpha[Alpha<1e-3]=0
print(np.nonzero(Alpha))


# In[24]:


w0=(1/y[469])-np.dot(w.T,X[469].reshape(2,1))
print(w0)


# In[29]:


# prediction
test=np.array([[2,0.5],[0.8,0.7],[1.58,1.33],[0.008, 0.001]])
for elem in test:
    if np.dot(w.T,elem)+w0>0:
        print("The class is 1")
    else:
        print("The class is -1")

