import numpy as np
import cv2
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


f=open('t10k-images-idx3-ubyte','rb')
#f=open('train-images-idx3-ubyte','rb')
mno=int.from_bytes(f.read(4),byteorder='big')
d1=int.from_bytes(f.read(4),byteorder='big')
d2=int.from_bytes(f.read(4),byteorder='big')
d3=int.from_bytes(f.read(4),byteorder='big')
   
buf = f.read(d1*d2*d3)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
data = data.reshape(d1,d2*d3) 
f.close()   

f=open('test_labels','r')
y=np.zeros((d1,10),dtype=np.float64)
for i in range(0,d1):
	y[i][int(f.readline())]=1
f.close()

w1=arrayfromfile('mat1.txt',28,(d2*d3)+1)
w2=arrayfromfile('mat2.txt',10,29)


l1=np.zeros(((d2*d3)+1),dtype=np.float64)
l2=np.zeros((29),dtype=np.float64)
l3=np.zeros((10),dtype=np.float64)


def sigmoid(arrone):
     n=arrone.shape[0]
     for i in range(0,n):
         arrone[i]=(1 / (1 + np.exp(-arrone[i])))
     return (arrone.copy())

def forwardprop(i):
    global data,w1,w2,w3,l1,l2,l3,l4
    l1[1:l1.shape[0]]=data[i].copy()
    l1[0]=1
    l1[1:l1.shape[0]]=l1[1:l1.shape[0]]*(1/255)
    l2[1:l2.shape[0]]=sigmoid(w1.dot(l1))
    l2[0]=1
    l3=sigmoid(w2.dot(l2))
    
choice=1
tstno=0
point=0
for tstno in range(0,d1):
    #print(l3)
    forwardprop(tstno)
    #print(l3,sep='\n')
    val=l3[0]
    key=0
    for xx in range(1,10):
        if l3[xx]>val:
            val=l3[xx]
            key=xx
    #cv2.imshow('k',data[tstno].reshape(d2,d3))
    ans=np.where(y[tstno]==1)
    ans=ans[0][0]
    if(ans==key):
        point+=1
   # print('ans',key,'actual',ans)
    #cv2.waitKey(0)

    #choice=int(input())
    if choice!=1:
        break
    tstno+=1
#cv2.destroyAllWindows()
print('performance',((point/d1)*100))
