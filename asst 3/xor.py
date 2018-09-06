import numpy as np

def generate_input(X,y,n):
    x1=[]
    y1=[]
    for i in range(0,4):
        for j in range(0,n):
            noise=np.random.normal(0,0.1,size=(2,1))
            x1.append(X[i].reshape(2,1)+noise)
            y1.append(y[i]+np.random.normal(0,0.1))
    x1=np.array(x1)
    y1=np.array(y1)
    return x1,y1

# x being the input of dimension 2x1, w being the weight matrix of dimension 2xN and b being the bias vector with dimension Nx1

def layer(w,x,b):
    out = np.matmul(w.T,x.reshape(len(x),1))+b
    return out
# the sigmoid function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# derivative of sigmoid function
def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
# derivative of sum of square error function
def mse(y_true,y_pred):
    return (y_true-y_pred)**2

X=np.array([[1,0],[0,1],[1,1],[0,0]])
y=np.array([1,1,0,0])
# initialisation of the weights
number_train_samples = int(raw_input("Enter the number of training samples: "))
number_nodes=int(raw_input("Enter the number of nodes in hidden layer: "))
learning_rate=float(raw_input("learning_rate: "))

X1=generate_input(X,y,number_train_samples)
X=X1[0]
y=X1[1]
X=X.reshape(len(y),2)
print(X.shape)
print(y.shape)



W1=np.random.normal(0,1,(2,number_nodes))
W2=np.random.normal(0,1,(number_nodes,1))
b1=np.random.normal(0,1,(number_nodes,1))
b2=np.random.normal(0,1,(1,1))

# iterative back propagation
epochs=100

for k in range(0,epochs):
    total_1 = np.zeros(W1.shape)
    total_2 = np.zeros(W2.shape)
    bias_1 = np.zeros(b1.shape)
    bias_2 = np.zeros(b2.shape)
    for i in range(0,len(y)):
        # forward pass
        out_1 = layer(W1,X[i],b1)
        z = sigmoid(out_1)
        out_2 = layer(W2,z,b2)
        y_pred= sigmoid(out_2)
        loss = mse(y[i],y_pred)
        # back propagation
        bias_2 =bias_2+2*(y_pred-y[i])*derivative_sigmoid(out_2)
        total_2=total_2+2*(y_pred-y[i])*derivative_sigmoid(out_2)*z
        for j in range(0,number_nodes):
            bias_1[j]=bias_1[j]+2*(y_pred-y[i])*derivative_sigmoid(out_2)*W2[j]*derivative_sigmoid(out_1[j])
            total_1[:,j]=total_1[:,j]+2*(y_pred-y[i])*derivative_sigmoid(out_2)*W2[j]*derivative_sigmoid(out_1[j])*X[i]
    print(loss)
    b2=b2-learning_rate*bias_2
    b1=b1-learning_rate*bias_1
    W2=W2-learning_rate*total_2
    W1=W1-learning_rate*total_1

while(1):
    a=raw_input("Enter test sample: ").split(',')
    for i in range(0,2):
        a[i]=float(a[i])
    a=np.array(a)
    print(a.shape)
    out_1 = layer(W1,a,b1)
    z = sigmoid(out_1)
    out_2 = layer(W2,z,b2)
    y_pred= sigmoid(out_2)
    print(y_pred)
