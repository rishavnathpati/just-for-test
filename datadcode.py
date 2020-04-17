import cv2
import numpy as np

#labels
#renmae to change the required label file to change it into normal text file(for self checking)
f=open('train-labels-idx1-ubyte','rb')
p=open('train_labels','w+')
'''[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
'''
lmno=int.from_bytes(f.read(4),byteorder='big')
ld1=int.from_bytes(f.read(4),byteorder='big')

buf = f.read(ld1)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
labels = labels.reshape(ld1, 1) 
    
f.close()         

for i in range(0,ld1):
	c=int(labels[i])
	p.write(str(c))
	p.write('\n')
p.close()


