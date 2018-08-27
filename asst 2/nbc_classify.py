import csv
import numpy as    np

# reads input from csv files
def read_data(filename):
    X=[]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            X.append(np.array(row).astype('float64'))
    X=np.array(X)
    return X

# returns the gaussian pdf value
def distribution(mean,variance,x):
    exponent=[0,0]
    exponent[0]=(x[0]-mean[0])**2
    exponent[0]=exponent[0]/(2.0*variance[0])
    exponent[1]=(x[1]-mean[1])**2
    exponent[1]=exponent[1]/(2.0*variance[1])
    exponent1=exponent[0]+exponent[1]
    cons=2*np.pi*np.sqrt(variance[0]*variance[1])
    cons=1.0/cons
    return cons*np.exp(-(exponent1))


X=read_data('X.csv')
y=read_data('Y.csv')
X=X.T
y=y.astype('int')
y=y.reshape(1000,)
df={'1':[],'-1':[]}

for i in range(0,len(X)):
    df[str(y[i])].append(X[i])

pr_y_1=len(df['1'])*1.0/len(y) # probability that y=1
pr_y_2=1-pr_y_1  # probability that y=-1

df['1']=np.array(df['1'])
df['-1']=np.array(df['-1'])

mean_y_1=np.mean(df['1'],axis=0)
var_y_1=np.var(df['1'],axis=0)

mean_y_2=np.mean(df['-1'],axis=0)
var_y_2=np.var(df['-1'],axis=0)

#### Testing part ####
x_test=[0,0]
x_test[0]=int(raw_input("Enter the first component of test sample: "))
x_test[1]=int(raw_input("Enter the first component of test sample: "))

pr_class_1=distribution(mean_y_1,var_y_1,x_test)*pr_y_1
pr_class_2=distribution(mean_y_2,var_y_2,x_test)*pr_y_2
print(pr_class_1)
print(pr_class_2)
if pr_class_1 > pr_class_2:
    print("It belongs to label 1")
else:
    print("It belongs to label -1")
