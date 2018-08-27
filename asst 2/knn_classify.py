import csv
import numpy as np

def read_data(filename):
    X=[]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            X.append(np.array(row).astype('float64'))
    X=np.array(X)
    return X

def bubble_sort(a):
    n=len(a)
    for i in range(0,n):
        for j in range(0,n-i-1):
            if a[j]>a[j+1]:
                temp=a[j]
                a[j]=a[j+1]
                a[j+1]=temp
    return a


def distance(X,Y):
    return (np.sum((X-Y)**2))**(0.5)

def classifier(X,y,x_test,k):
    #refer=np.vstack((df['1'],df['-1']))
    dist=[]
    pair=[]
    x_test=np.array(x_test)
    for elem in X:
        dist.append(distance(elem,x_test))
    dist=np.array(dist)
    for i in range(0,len(y)):
        pair.append((dist[i],y[i]))
    pair=bubble_sort(pair)
    flag=0
    for i in range(0,k):
        flag=flag+pair[i][1]
    return flag


X=read_data('X.csv')
y=read_data('Y.csv')
X=X.T
print(X.shape)
y=y.astype('int')
x_test=[0,0]
x_test[0]=int(raw_input("Enter the first component of test sample: "))
x_test[1]=int(raw_input("Enter the first component of test sample: "))
k=int(raw_input("Enter the number of neighbors: "))
k1=classifier(X,y,x_test,k)
#print(k1)
if k1>0:
    print("Belongs to label 1")
elif k1<0:
    print("Belongs to label -1")
else:
    print("Could be both classes")
