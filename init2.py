import numpy as np
import numpy.random as rnd 

#this will randomly initialize weight matrices(the input will be the dimention of the weight matrix).....here i used a nural network with 785(28*28),29,10 nodes in its 1st,2nd and 3rd layer respectively,considering the bias terms
n=int(input("enter no of weight matrices "))
for number in range(1,n+1):
    f=open('mat'+str(number)+'.txt','w+')
    d1=int(input('dimension 1 for matrix'+str(number)))
    d2=int(input('dimension 2 for matrix'+str(number)))
    wh=rnd.randint(100,size=(d1,d2))
    wh=wh/100
    for i in range(0,d1):
        for j in range(0,d2):
            c=wh[i,j]-0.5
            f.write(str(c))
            f.write('\n')
    f.close()


       
