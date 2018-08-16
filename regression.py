import numpy as np 
from matplotlib import pyplot as plt 

def create_data_matrix(X,N,d):
	length=len(X)
	X=X.reshape(length,1)
	X=np.hstack(X**i for i in range(0,d+1)) 
	return X

def Regression(X,y,l):
	X_hat=np.matmul(X.T,X)
	identity=np.identity(len(X_hat))
	identity[0][0]=0
	X_hat=X_hat+l*np.identity(len(X_hat))
	X_hat=np.matmul(np.linalg.inv(X_hat),X.T)
	return np.matmul(X_hat,y)

# all user inputs
print("Enter number of training samples: ")
N=int(raw_input())
print("Enter the degree of polynomial: ")
d=int(raw_input())
print("Enter Lagrangian multiplier: ")
l=float(raw_input())
print("Enter number of test samples: ")
N_test=int(raw_input())

#training part
x=np.linspace(0,2*np.pi,N)
mean=0
std=0.05
y=np.sin(x)+np.random.normal(mean,std,N)
y=y.reshape(N,1)
X_train=create_data_matrix(x,N,d)
w=Regression(X_train,y,l)

# testing part
x_test=np.linspace(0,2*np.pi,N_test)
y_true=np.sin(x_test)
x_test1=create_data_matrix(x_test,N_test,d)
y_pred=np.matmul(x_test1,w)


plt.plot(x_test,y_true,'x')
plt.plot(x_test,y_pred)
plt.grid()
plt.show()
