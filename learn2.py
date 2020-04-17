import cv2
import numpy as np

#image

def arrayfromfile(fname,dim1,dim2):
    f=open(fname,'r+')
    a=[]
    for i in range(0,dim1):
        a.append([])
        for j in range(0,dim2):
            c=f.readline()
            a[i].append(c)
    f.close()
    return(np.array(a,dtype=np.float64))
    
#reading the image from dataset

f=open('train-images-idx3-ubyte','rb')

mno=int.from_bytes(f.read(4),byteorder='big')
d1=int.from_bytes(f.read(4),byteorder='big')
d2=int.from_bytes(f.read(4),byteorder='big')
d3=int.from_bytes(f.read(4),byteorder='big')
   
buf = f.read(d1*d2*d3)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
data = data.reshape(d1,d2*d3) 
    
f.close()         

#reading answers (it is a text file which is created based on mnist dataset of training labels 
#the y array is in vector form (if the answer is 7 then only y[7] will be 1 and all other position wil contain 0.and this will happen for all the examples
f=open('train_labels','r+')
y=np.zeros((d1,10),dtype=np.float64)
for i in range(0,d1):
	y[i][int(f.readline())]=1
f.close()

#here the weight matrices are initialized.the random initialization was saved on mat1.txt and mat2.txt for matrix from 1st layer to 2nd layer and 2nd layer to 3rd layer(output layer) respectively
#you have to change the dimention of these weight matrices to use a network with different no. of nodes
w1=arrayfromfile('mat1.txt',28,(d2*d3)+1)
w2=arrayfromfile('mat2.txt',10,29)

#all layers are initialized with zero

l1=np.zeros(((d2*d3)+1),dtype=np.float64)
l2=np.zeros((29),dtype=np.float64)
l3=np.zeros((10),dtype=np.float64)

#the sigmoid funtion can also be written as vector way as it will cause no problem when using numpy.array

def sigmoid(arrone):
     n=arrone.shape[0]
     for i in range(0,n):
         arrone[i]=(1 / (1 + np.exp(-arrone[i])))
     return (arrone.copy())
#code for forward propagation.//parameter:-the index of traing set on which fordward-prop has to be applied
def forwardprop(i):
    global data,w1,w2,l1,l2,l3
    l1[1:l1.shape[0]]=data[i].copy()
    l1[0]=1
    l1[1:l1.shape[0]]=l1[1:l1.shape[0]]*(1/255)
    l2[1:l2.shape[0]]=sigmoid(w1.dot(l1))
    l2[0]=1
    l3=sigmoid(w2.dot(l2))
    
#larning rate
alpha=1
#arrays for error calculation
delta1=np.zeros((28,(d2*d3)+1),dtype=np.float64)
delta2=np.zeros((10,29),dtype=np.float64)


del2=l2.copy()
del3=l3.copy()


avgchange=100
totalitem=0
lmda=1
iterxxx=0
#no of trainig examples to train with
ttltrain=10000
totalitem+=delta1.shape[0]*delta1.shape[1]
totalitem+=delta2.shape[0]*delta2.shape[1]
preverr=np.zeros((10),dtype=np.float64)
curerr=np.zeros((10),dtype=np.float64)
err=np.zeros((10),dtype=np.float64)
err[0]=2
#the iterxxx is given for controlling the no of iteration
while iterxxx<300:
    iterxxx+=1
    for i in range(0,ttltrain):
        curerr=np.zeros((10),dtype=np.float64)
        forwardprop(i)
        del3=(y[i]-l3)*(l3*(1-l3))
        curerr+=((y[i]-l3)*(y[i]-l3))
        del2=(w2.transpose().dot(del3))*(l2*(np.ones(l2.shape)-l2))
        delta1=delta1+(del2[1:29].reshape(del2[1:29].shape[0],1).dot(l1.reshape(1,l1.shape[0])))
        delta2=delta2+(del3.reshape(del3.shape[0],1).dot(l2.reshape(1,l2.shape[0])))
    curerr=curerr/(2*ttltrain)
    err=curerr-preverr
    if iterxxx!=1 and curerr.sum()>=preverr.sum():
        break;
    preverr=curerr.copy()
    delta1=delta1*(alpha/ttltrain)
    delta2=delta2*(alpha/ttltrain)
    te1=w1+delta1
    te2=w2+delta2
    
    te1[:,0]= te1[:,0]+(w1[:,0]*(lmda/ttltrain))
    te2[:,0]= te2[:,0]+(w2[:,0]*(lmda/ttltrain))
    w1=te1.copy()
    w2=te2.copy()
    delta1=np.zeros((28,(d2*d3)+1),dtype=np.float64)
    delta2=np.zeros((10,29),dtype=np.float64)
    print('backprop',i,curerr.sum(),iterxxx)

#saving all the changed weights after traing into corresponding files
f=open('mat1.txt','w')
for chgol in range(0,w1.shape[0]):
    for goru in range(0,w1.shape[1]):
        c=w1[chgol,goru]
        f.write(str(c))
        f.write('\n')
f.close()
f=open('mat2.txt','w')
for chgol in range(0,w2.shape[0]):
    for goru in range(0,w2.shape[1]):
        c=w2[chgol,goru]
        f.write(str(c))
        f.write('\n')
f.close()


















