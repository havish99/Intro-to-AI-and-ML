import numpy as np 
from matplotlib import pyplot as plt 

def create_data_matrix(X,d):
	length=len(X)
	X=X.reshape(length,1)
	X=np.hstack(X**i for i in range(0,d+1)) 
	return X

def Regression(X,y):
	X_hat=np.matmul(X.T,X)
	X_hat=np.matmul(np.linalg.inv(X_hat),X.T)
	return np.matmul(X_hat,y)

# all user inputs
print("Enter number of training samples: ")
N=int(raw_input())
print("Enter the degree of polynomial: ")
d=int(raw_input())
print("Enter number of test samples: ")
N_test=int(raw_input())
print("Enter variance of labels: ")
var=float(raw_input())

#training part
x=np.linspace(0,2*np.pi,N)
mean=0
std=0.05
y=np.sin(x)+np.random.normal(mean,std,N)
y=y.reshape(N,1)
X_train=create_data_matrix(x,d)
print(X_train)
w=Regression(X_train,y)


# testing part
x_test=np.linspace(0,2*np.pi,N_test)
#x_test=x_test+np.random.normal(mean,std,N_test)
y_true=np.sin(x_test)
x_test1=create_data_matrix(x_test,d)
y_pred=[np.random.normal(np.matmul(x_test1[i],w),np.sqrt(var)) for i in range(0,len(x_test1))] 
y_pred=np.array(y_pred)

plt.plot(x_test,y_true,'x',label='True values')
plt.plot(x_test,y_pred,'o',label='Prediction')
plt.grid()
plt.legend()
plt.savefig('variance.png')
plt.show()
