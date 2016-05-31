# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:00:43 2016

@author: dtward
"""

import NN
import numpy as np
import matplotlib.pyplot as plt



# at this point I should have enough to run a test
# here's a very simple neural net
# two inputs
sigmaR = 10.0
#sigmaR = None
nn = NN.NeuralNet([
                #input layer
                NN.Matrix([2,1],[5,1],sigmaR=sigmaR),
                NN.Addition([5,1],sigmaR=sigmaR),
                NN.ComponentwiseFunction(),
                #interneuron
                NN.Matrix([5,1],[5,1],sigmaR=sigmaR),
                NN.Addition([5,1],sigmaR=sigmaR),
                NN.ComponentwiseFunction(),
                # interneuron 2
                NN.Matrix([5,1],[5,1],sigmaR=sigmaR),
                NN.Addition([5,1],sigmaR=sigmaR),
                NN.ComponentwiseFunction(),
                # output
                #Matrix([5,1],[2,1],sigmaR=sigmaR),
                #Addition([2,1],sigmaR=sigmaR) 
                NN.Matrix([5,1],[3,1],sigmaR=sigmaR),
                NN.Addition([3,1],sigmaR=sigmaR) 
                ])
# a nice simple interface
nn = NN.makeStandardNeuralNet(inputDim=2,outputDim=3,interDim=20,nInter=5,sigmaR=sigmaR)
# simple training set in 2D
n = 100
x = np.zeros([2,1,n])
z = np.zeros([2,1,n])
for i in range(n):
    off = np.random.rand() > 0.5
    x[:,:,i] = np.random.randn(2,1) + off*3
    z[0,:,i] = float(off)
    z[1,:,i] = 1.0 - float(off)
    
n = 200
x = np.zeros([2,1,n])
z = np.zeros([3,1,n])
for i in range(n):
    category = np.random.randint(3)
    x[:,:,i] = np.random.randn(2,1) \
       + (category==0)*np.array([[-1],[-1]])*3*((np.random.rand()>0.5)-0.5)*2.0 \
       + (category==1)*np.array([[1],[0]])*2.5 \
       + (category==2)*np.array([[0],[1]])*2.5
    z[0,:,i] = float(category==0)
    z[1,:,i] = float(category==1)   
    z[2,:,i] = float(category==2)   
    
    
    

plt.close('all')
hFig = plt.figure()
hAxScatter = hFig.add_subplot(1,2,1)
hScatter = hAxScatter.scatter(x[0,:],x[1,:])

nn.setTrainingData(x,z)
nn(x[:,:,0])



for ii in range(100):
    nn.train(10,0.001)
    nImage = [50,50]
    I=np.zeros(nImage)
    J=np.zeros(nImage)
    K=np.zeros(nImage)
    xdomain = np.linspace(-5,8,nImage[1])
    ydomain = np.linspace(-5,8,nImage[0])
    for i in range(nImage[1]):
        for j in range(nImage[0]):
            out = nn(np.array([[xdomain[j]],[ydomain[i]]]))
            I[i,j] = out[0][0]
            J[i,j] = out[1][0]
            K[i,j] = out[2][0]
    hFig.clear()
    #hAx1 = hFig.add_subplot(1,2,1)
    hAx1 = hFig.add_subplot(1,3,1)
    hScatter1 = hAx1.scatter(x[0,:,z[0,:,:][0]==1],x[1,:,z[0,:,:][0]==1],c='r')
    hScatter11 = hAx1.scatter(x[0,:,z[1,:,:][0]==1],x[1,:,z[1,:,:][0]==1],c='b')
    hScatter12 = hAx1.scatter(x[0,:,z[2,:,:][0]==1],x[1,:,z[2,:,:][0]==1],c='g')
    hIm = hAx1.imshow(I,extent=(xdomain[0],xdomain[-1],ydomain[0],ydomain[-1]),origin='bottom')
    plt.colorbar(hIm)
    
    #hAx2 = hFig.add_subplot(1,2,2)
    hAx2 = hFig.add_subplot(1,3,2)
    hScatter2 = hAx2.scatter(x[0,:,z[0,:,:][0]==1],x[1,:,z[0,:,:][0]==1],c='r')
    hScatter21 = hAx2.scatter(x[0,:,z[1,:,:][0]==1],x[1,:,z[1,:,:][0]==1],c='b')
    hScatter22 = hAx2.scatter(x[0,:,z[2,:,:][0]==1],x[1,:,z[2,:,:][0]==1],c='g')
    hIm2 = hAx2.imshow(J,extent=(xdomain[0],xdomain[-1],ydomain[0],ydomain[-1]),origin='bottom')
    plt.colorbar(hIm2)
    
    hAx3 = hFig.add_subplot(1,3,3)
    hScatter3 = hAx3.scatter(x[0,:,z[0,:,:][0]==1],x[1,:,z[0,:,:][0]==1],c='r')
    hScatter31 = hAx3.scatter(x[0,:,z[1,:,:][0]==1],x[1,:,z[1,:,:][0]==1],c='b')
    hScatter32 = hAx3.scatter(x[0,:,z[2,:,:][0]==1],x[1,:,z[2,:,:][0]==1],c='g')
    hIm3 = hAx3.imshow(K,extent=(xdomain[0],xdomain[-1],ydomain[0],ydomain[-1]),origin='bottom')
    plt.colorbar(hIm3)
    
    plt.pause(0.1)

raise Exception


# really simple example, just multiply
# currently it works without a regularization, but not with one, at least not as expected
# I should definitely get a factor closer to zero when I use regularization
# seems to be working now
nn = NN.NeuralNet([NN.Multiplication([1],sigmaR=10.0)])
x = np.array(range(10))
x.shape=[1,x.shape[0]]
factor = 10.0
z = x*factor
initialFactor = nn.layers[0].parameters
nn.setTrainingData(x,z)
nn.train(1,0.001)
print('initial factor: {0}'.format(initialFactor))
print('final factor: {0}'.format(nn.layers[0].parameters))
print('true factor: {0}'.format(factor))


# try for shift
nn = NN.NeuralNet([NN.Addition([1],sigmaR=10.0)])
x = np.array(range(10))
x.shape=[1,x.shape[0]]
factor = 10.0
z = x+factor
initialFactor = nn.layers[0].parameters
nn.setTrainingData(x,z)
nn.train(1,0.0001)
print('initial factor: {0}'.format(initialFactor))
print('final factor: {0}'.format(nn.layers[0].parameters))
print('true factor: {0}'.format(factor))
# seems to be working


nn = NN.NeuralNet([NN.Multiplication([1],sigmaR=100.0),NN.Addition([1],sigmaR=100.0)])
x = np.array(range(20))
x.shape=[1,x.shape[0]]
factor = 10.0
constant = 4.0
z = factor*x+constant
initialFactor = nn.layers[0].parameters
initialConstant = nn.layers[1].parameters
nn.setTrainingData(x,z)
nn.train(1,0.0001)
print('initial factor: {0}'.format(initialFactor))
print('initial constant: {0}'.format(initialConstant))
print('final factor: {0}'.format(nn.layers[0].parameters))
print('final constant: {0}'.format(nn.layers[1].parameters))
print('true factor: {0}'.format(factor))
print('true constant: {0}'.format(constant))
# seems to be working


# try for matrix multiplication, scalar case first
nn = NN.NeuralNet([NN.Matrix([1],[1],sigmaR=10.0)])
x = np.array(range(10),dtype='float')
x.shape=[1,x.shape[0]]
factor = -0.5
z = x*factor
initialFactor = nn.layers[0].parameters
nn.setTrainingData(x,z)
nn.train(1,0.001)
print('initial factor: {0}'.format(initialFactor))
print('final factor: {0}'.format(nn.layers[0].parameters))
print('true factor: {0}'.format(factor))

# for non scalar
nn = NN.NeuralNet([NN.Matrix([2,1],[2,1],sigmaR=None), NN.Addition([2,1],sigmaR=None) ])
#nn = NeuralNet([Matrix([2,1],[2,1],sigmaR=None)])
x = np.array(range(14),dtype='float')
x.shape=(2,1,7)
A = np.array([[1.0,0.5],[0.75,2.0]])
b = np.array([[1.0,1.0]]).transpose()
z = np.array(x)
for i in range(x.shape[-1]):
    z[:,:,i] = A.dot(x[:,:,i]) + b
    #z[:,:,i] = A.dot(x[:,:,i]) 
nn.setTrainingData(x,z)
nn.train(10000,0.001)
print( nn[0].parameters )
print( nn[1].parameters )
nn(x[:,:,0])
z[:,:,0]
z[:,:,0] - nn(x[:,:,0])
# well, the gradient goes to zero, but I don't find the right result
x.shape = (2,7)
z.shape = (2,7)
#a = np.linalg.solve(x,z)
# A x = z
# A x x' = z x'
# A = z x' (x x ')^{-1}
a = z.dot(x.transpose()).dot(  np.linalg.inv( x.dot(x.transpose()) )  )
# well the problem was integers!

# okay the interesting thing is that these two are equal!
# although neither of them are my original A
# guh, fixed, integer vs float