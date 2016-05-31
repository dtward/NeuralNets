# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:47:11 2016

@author: dtward


This will be classes to make general neural nets.

Each layer will need these things.

1. Set parameters / get parameters

2. Update parameters based on error

3. Apply

4. Apply adjoint

5. Dimension of input and dimension of output

6. reg energy and gradient

In particular I will want certain types of layers

1. squashing function.  no parameters.  adjoint is multiplication with derivative.  Generally other layers will be linear

2. Linear layer.  Matrix A

3. Shift layer.  vector b

4. Scale layer.  Multiplier a

3. Convolution layer.  A kernel as parameters

4. Downsample layer.  Particularly for downsampling images

5. Upsample layer.  For upsampling images, but more likely for making copies of images. I maybe want to call this a copy layer.

Design choice.  Since I intend to work with images, I'll make my data generally have size nrow,ncol,nclice,n for n images

"""

import numpy as np
import scipy.signal as sps


class Layer(object):    
    def __init__(self,xShape=None,yShape=None,parameters=None, sigmaR=None):
        ''' init signature is xshape, yshape, parameters, sigmaR
        these default to None, but generally xshape and yshape are required
        '''
        # the input dimension should be an iterable
        self.xShape = xShape
        self.x = [0]
        self.ex = [0]
        # the output dimension should be an iterable
        self.yShape = yShape
        self.y = [0]
        self.ey = [0]
        # the parameters
        self.parameters = [0]
        self.gradient = [0]
        print("Instantiating a neural network layer base class")
    
    def applyLayer(self):
        ''' needs to be implemented '''
        self.y = self.x
        
    def applyAdjoint(self):
        ''' needs to be implemented '''
        self.ex = self.ey
    
    def incrementGradient(self):
        ''' needs to be implemented '''
        pass
    
    def zeroGradient(self):
        ''' default '''
        if self.gradient is not None:
            self.gradient *= 0.0        
    
    def updateParameters(self,epsilon):
        ''' default '''
        if self.gradient is not None:
            self.parameters = self.parameters - epsilon * self.gradient
    
    def incrementGradientWithRegularization(self):
        ''' default '''
        if self.sigmaR is not None:
            self.gradient += (self.parameters/(self.sigmaR**2))
    
    def calculateRegularizationEnergy(self):
        ''' default '''
        self.regularizationEnergy = 0
        if self.sigmaR is not None:
            for p in np.nditer(self.parameters):
                self.regularizationEnergy += p**2    
            self.regularizationEnergy /= 2.0*self.sigmaR**2
            
    def setX(self,x):
        # this function sets x and checks the shape
        if x.shape != self.xShape:
            raise Exception('Attempting to set x with shape {}, but expecting shape {}'.format(x.shape,self.xShape))
        self.x = x
            
    def __repr__(self):
        return "(Base Layer)"
        
    
    
class NeuralNet:
    def __init__(self,layers=None,parameters=(lambda y,z: 0.5*np.sum(np.power(y-z,2)), lambda y,z: y-z )):
        ''' The input should be an interable of layers '''
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.checkSizes()
        self.loss = parameters[0]
        self.lossPrime = parameters[1]
        
    def addLayer(self,layer):
        self.layers.append(layer)
        self.checkSizes()
        
    def checkSizes(self):
        ''' for checking sizes, I'd like to skip ones with a None for their shape '''
        layersCheckInput = filter(lambda x: x.xShape is not None, self.layers[1:])
        layersCheckOutput = filter(lambda x: x.yShape is not None, self.layers[:-1])
        for i,layers in enumerate(zip(layersCheckOutput,layersCheckInput)):
            layer0,layer1 = layers            
            # none will mean it accepts any size
            if tuple(layer0.yShape) != tuple(layer1.xShape):
                raise Exception("{0} ({2}) output and {1} ({3}) input sizes do not match".format(layer0, layer1, i, i+1))
                
    def applyNeuralNetwork(self,inputData):
        # no checks here because I want it to be fast
        self.layers[0].x = inputData
        for i in xrange(len(self.layers)-1):
            self.layers[i].applyLayer()
            self.layers[i+1].x = self.layers[i].y
        self.layers[-1].applyLayer()
        return self.layers[-1].y     
        
    def __call__(self,inputData):
        return self.applyNeuralNetwork(inputData)
        
    def __getitem__(self,i):
        return self.layers[i]
        
    def __iter__(self):
        # return something with a next method, can be yourself
        return iter(self.layers)
    
    def __len__(self):
        return len(self.layers)
        
    def backpropogate(self,error):
        # no checks because I want to be fast
        self.layers[-1].ey = error
        for i in xrange(len(self.layers)-1,0,-1):
            self.layers[i].incrementGradient()
            self.layers[i].applyAdjoint()
            self.layers[i-1].ey = self.layers[i].ex
        self.layers[0].incrementGradient()
        self.layers[0].applyAdjoint()
            
    def setTrainingData(self,trainingInput, trainingOutput):
        self.trainingInput = trainingInput
        self.trainingOutput = trainingOutput
        layersCheckInput = filter(lambda x: x.xShape is not None, self.layers)
        layersCheckOutput = filter(lambda x: x.yShape is not None, self.layers)
        if tuple(layersCheckInput[0].xShape) != tuple(self.trainingInput.shape[:-1]):
            raise Exception("Input data is the wrong shape")
        if tuple(layersCheckOutput[-1].yShape) != tuple(self.trainingOutput.shape[:-1]):
            raise Exception("Output data is the wrong shape")
            
    # I'll use square error loss until I decide to change it later
    def train(self,maxIter=100,epsilon=0.01):
        LOld = 1e20 
        for it in xrange(maxIter):
            # initialize loss
            self.E = 0.0
            self.R = 0.0
            self.L = 0.0
            # initialize gradient
            for layer in self.layers:
                # this type of iteration should be okay, I should be able to modify
                layer.zeroGradient()
                layer.calculateRegularizationEnergy()
                self.R += layer.regularizationEnergy
                layer.incrementGradientWithRegularization()
            print('R is {}'.format(self.R))
            
            for x,z in zip(np.rollaxis(self.trainingInput, -1), np.rollaxis(self.trainingOutput, -1)):
                y = self.applyNeuralNetwork(x)
                self.E += self.loss(y,z)
                self.backpropogate(self.lossPrime(y,z))
            for layer in self.layers:
                layer.updateParameters(epsilon)
            print('Error is {}'.format(self.E))
            self.L = self.E + self.R
            print('Loss is {}'.format(self.L))
            #print('LOld is {}'.format(LOld))
            if self.L > LOld:
                #print('LOld < self.L')
                #for layer in self.layers:
                #    layer.updateParameters(-epsilon)
                #epsilon *= 0.9
                #print('reducing epsilon to {}'.format(epsilon))
                pass
            else:
                LOld = self.L
            
        
            
    def __repr__(self):
        return "(Neural network {0}->{1} with {2} layers:\n {3})".format(self.layers[0].xShape, self.layers[-1].yShape,len(self.layers),self.layers)
        

# okay let's start simple!
class ComponentwiseFunction(Layer):
    ''' apply the same function to every value, defaults to atan '''
    def __init__(
            self, xShape=None, yShape=None, 
            parameters=(lambda x: np.arctan(x), lambda x: np.power(1.0 + np.power(x,2.0),-1.0)),
            sigmaR=None ):
        # the parameters should be f and f prime
        self.parameters = parameters
        # this layer doesn't care about dimensions.  Numpy will take care of that
        self.xShape = xShape
        self.yShape = yShape
        self.sigmaR  = None
        self.gradient = None
        
    def applyLayer(self):
        self.y = self.parameters[0](self.x)
        
    def applyAdjoint(self):
        self.ex = self.parameters[1](self.x)*self.ey
        
    def __repr__(self):
        return "(ComponentwiseFunction Layer)"
        
    # nothing else is necessary

class Matrix(Layer):
    def __init__(self, xShape, yShape, parameters=None, sigmaR=None):
        self.xShape = xShape
        self.yShape = yShape
        # matrix
        self.nRows = np.prod(self.yShape)
        self.nCols = np.prod(self.xShape)
        if parameters is None:
            self.parameters = np.random.randn(self.nRows,self.nCols)*0.1 + np.eye(self.nRows,self.nCols)
        else:
            self.parameters = parameters
        self.sigmaR = sigmaR 
        self.gradient = np.zeros(self.parameters.shape)
        
    def applyLayer(self):
        x_ = self.x.ravel().transpose()
        y_ = self.parameters.dot(x_)
        self.y = np.reshape(y_, self.yShape)
        
    def applyAdjoint(self):
        ey_ = self.ey.ravel().transpose()
        ex_ = self.parameters.transpose().dot(ey_)
        self.ex = np.reshape(ex_, self.xShape)
    
    def incrementGradient(self):
        # for a matrix, the gradient is simple
        self.gradient += np.reshape( self.ey , (self.nRows,1) ).dot(np.reshape( self.x, (1,self.nCols) ) ) 
        
    def __repr__(self):
        return "(Matrix Layer {0}->{1})".format(self.xShape,self.yShape)
        
class Addition(Layer):
    def __init__(self, xShape, yShape=None, parameters=None, sigmaR=None):
        self.xShape = xShape
        self.yShape = yShape
        if self.yShape is None:
            self.yShape = self.xShape
        if self.xShape != self.yShape:
            raise Exception("For an Addition layer, xShape and yShape, must be equal")
        # vector
        if parameters is None:
            self.parameters = np.random.randn(*self.xShape)*0.01
        else:
            self.parameters = parameters
        self.sigmaR  = sigmaR
        self.gradient = np.zeros(self.parameters.shape)
        
    def applyLayer(self):
        self.y = self.x + self.parameters
        
    def applyAdjoint(self):
        self.ex = self.ey
    
    def incrementGradient(self):
        # for a matrix, the gradient is simple
        self.gradient += self.ey
        
    def __repr__(self):
        return "(Addition Layer {0}->{1})".format(self.xShape,self.yShape)
        
class Multiplication(Layer):
    def __init__(self, xShape, yShape=None, parameters=None, sigmaR=None):
        self.xShape = xShape
        self.yShape = yShape
        if self.yShape is None:
            self.yShape = self.xShape
        if self.xShape != self.yShape:
            raise Exception("For a Multiplication layer, xShape and yShape, must be equal")
        # vector
        if parameters is None:
            self.parameters = np.random.randn(*self.xShape)*0.01 + 1.0
        else:
            self.parameters = parameters
        self.gradient = np.zeros(self.parameters.shape)
        self.sigmaR = sigmaR
        
    def applyLayer(self):
        self.y = self.x * self.parameters
        
    def applyAdjoint(self):
        self.ex = self.ey * self.parameters
    
    def incrementGradient(self):
        self.gradient += self.x * self.ey 
        
    def __repr__(self):
        return "(Multiplication Layer {0}->{1})".format(self.xShape,self.yShape)
    


def makeStandardNeuralNet(inputDim=1,outputDim=1,interDim=5,nInter=2,sigmaR=10.0):
    nn = NeuralNet()
    nn.addLayer(Matrix([inputDim,1],[inputDim,1],sigmaR=sigmaR))
    nn.addLayer(Addition([inputDim,1],sigmaR=sigmaR))
    nn.addLayer(ComponentwiseFunction())
    for i in range(nInter):
        if i == 0:
            nn.addLayer(Matrix([inputDim,1],[interDim,1],sigmaR=sigmaR))
        else:
            nn.addLayer(Matrix([interDim,1],[interDim,1],sigmaR=sigmaR))       
        nn.addLayer(Addition([interDim,1],sigmaR=sigmaR))
        nn.addLayer(ComponentwiseFunction())
    nn.addLayer(Matrix([interDim,1],[outputDim,1],sigmaR=sigmaR))
    nn.addLayer(Addition([outputDim,1],[outputDim,1],sigmaR=sigmaR))
    nn.addLayer(ComponentwiseFunction())
    nn.addLayer(Matrix([outputDim,1],[outputDim,1],sigmaR=sigmaR))
    nn.addLayer(Addition([outputDim,1],[outputDim,1],sigmaR=sigmaR))
    return nn
    
# working on this    
class DownsampleImage(Layer):
    def __init__(self, xShape, yShape, parameters=None, sigmaR=None):
        self.xShape = xShape
        self.yShape = yShape
        # we infer downsampling from shapes
        # start with the first pixel
        # assume x = d y + r
        self.r0 = self.xShape[0]%self.yShape[0]
        self.d0 = self.xShape[0]//self.yShape[0]
        self.r1 = self.xShape[1]%self.yShape[1]
        self.d1 = self.xShape[1]//self.yShape[1]
        # this above approach isn't really working
        # I'll need an array
        self.downsampleArray0 = np.array(np.floor( np.arange(self.yShape[0])/float(yShape[0])*xShape[0]), dtype=int)
        self.downsampleArray1 = np.array(np.floor( np.arange(self.yShape[1])/float(yShape[1])*xShape[1]), dtype=int)
        
        # note that downsampling factor d may be 1 if dimensions are not specified well
        self.parameters = (self.d0,self.d1,self.r0,self.r1)
        self.gradient = None
        self.sigmaR = None
        
    def applyLayer(self):
        ''' This function assumes a stack of rgb images '''        
        #self.y = self.x[self.d0-1:-self.r0-1:self.d0,self.d1-1:-self.r0-1:self.d1,:,:]
        y_ = self.x[self.downsampleArray0,:,:,:]
        self.y = y_[:,self.downsampleArray1,:,:]
    
    def applyAdjoint(self):
        self.ex = np.zeros(self.xShape)
        #self.ex[self.d0-1::self.d0,self.d1-1::self.d1,:,:]=self.ey
        ex_ = np.zeros((self.xShape[0],self.yShape[1],self.xShape[2],self.xShape[3]))
        ex_[self.downsampleArray0,:,:,:] = self.ey
        self.ex[:,self.downsampleArray1,:,:] = ex_
                
    def __repr__(self):
        return "(DownsampleImage Layer {0}->{1})".format(self.xShape,self.yShape)


'''
I = np.random.randn(256,259,3,1)
l = DownsampleImage(I.shape,(128,128,3,I.shape[-1]))
l.x = I
l.applyLayer()
J = l.y
close('all')
figure()
imshow(I[:,:,:,0])
figure()
imshow(J[:,:,:,0])
l.ey = J
l.applyAdjoint()
K = l.ex
figure()
imshow(K[:,:,:,0])
raise Exception
'''


class CopyImage(Layer):
    def __init__(self,xShape,yShape,parameters=None,sigmaR=None):
        self.xShape = xShape
        self.yShape = yShape
        # shapes should be the same except for last one
        if self.xShape[0] != self.yShape[0] or self.xShape[1] != self.yShape[1] or self.xShape[2] != self.yShape[2]:
            raise Exception('yShape {} should be the same as x shape {} except for the last dimension'.format(self.yShape,self.xShape))
        # now we will upsample
        # yshape = c xshape + r
        self.r = self.yShape[-1] % self.xShape[-1]
        self.c = self.yShape[-1] // self.xShape[-1]
        # I'll make c copies of everything, and r extra copies of the last layer
        # I want the largest value to map to the largest value of yshape
        arr = [ [i]*self.c for i in range(xShape[-1])]
        arr.append([xShape[-1]-1]*self.r)
        arr2 = [j for i in arr for j in i]
        self.copyArray = np.array(arr2)
        
        
        self.parameters = (self.r,self.c)
        
        self.gradient = None
        self.sigmaR = None

    def applyLayer(self):
        self.y = self.x[:,:,:,self.copyArray]
        
    def applyAdjoint(self):
        # let L be the copy layer
        # say I has size 1
        # okay sum_ijkl J_ijkl [LI]_ijkl = sum_ijk [L^*J]_ijk I_ijk
        # sum_ijk (sum_l J_ijkl) I_ijk
        # this shows we simply sum up J over the copies
        self.ex = np.zeros(self.xShape)
        for i in xrange(self.xShape[-1]):
            self.ex[:,:,:,i] = np.sum(self.ey[:,:,:,self.copyArray==i], axis=-1)
        
        
    def __repr__(self):
        return "(CopyImage Layer {0}->{1})".format(self.xShape,self.yShape)
        
        
'''
I = np.random.randn(256,259,3,2)
l = CopyImage(I.shape,[I.shape[0],I.shape[1],I.shape[2],3])
l.x = I
l.applyLayer()
J = l.y
close('all')
figure()
imshow(I[:,:,:,0])
figure()
imshow(J[:,:,:,0])
l.ey = J
l.applyAdjoint()
K = l.ex
figure()
imshow(K[:,:,:,0])

raise Exception
'''

class CombineImage(CopyImage):
    ''' This is exactly the same as CopyImage, but with adjoint and apply switched'''
    def __init__(self,xShape,yShape,parameters=None,sigmaR=None):
        self.xShape = xShape
        self.yShape = yShape
        # shapes should be the same except for last one
        if self.xShape[0] != self.yShape[0] or self.xShape[1] != self.yShape[1] or self.xShape[2] != self.yShape[2]:
            raise Exception('yShape {} should be the same as x shape {} except for the last dimension'.format(self.yShape,self.xShape))
        # now we downsample
        # xshape = c yshape + r
        self.r = self.xShape[-1] % self.yShape[-1]
        self.c = self.xShape[-1] // self.yShape[-1]
        # I'll make c copies of everything, and r extra copies of the last layer
        # I want the largest value to map to the largest value of yshape
        arr = [ [i]*self.c for i in range(xShape[-1])]
        arr.append([xShape[-1]-1]*self.r)
        arr2 = [j for i in arr for j in i]
        self.copyArray = np.array(arr2)
        
        self.parameters = (self.r,self.c)
        
        self.gradient = None
        self.sigmaR = None

    def applyAdjoint(self):
        self.ex = self.ey[:,:,:,self.copyArray]
        
    def applyLayer(self):
        # let L be the copy layer
        # say I has size 1
        # okay sum_ijkl J_ijkl [LI]_ijkl = sum_ijk [L^*J]_ijk I_ijk
        # sum_ijk (sum_l J_ijkl) I_ijk
        # this shows we simply sum up J over the copies
        self.y = np.zeros(self.yShape)
        for i in xrange(self.yShape[-1]):
            self.y[:,:,:,i] = np.sum(self.x[:,:,:,self.copyArray==i], axis=-1)
        
        
    def __repr__(self):
        return "(CombineImage Layer {0}->{1})".format(self.xShape,self.yShape)
        
'''        
I = np.random.randn(256,259,3,3)
l = CombineImage(I.shape,[I.shape[0],I.shape[1],I.shape[2],2])
l.x = I
l.applyLayer()
J = l.y
close('all')
figure()
imshow(I[:,:,:,0])
figure()
imshow(J[:,:,:,0])
l.ey = J
l.applyAdjoint()
K = l.ex
figure()
imshow(K[:,:,:,0])
raise Exception
'''

# last one!
class ConvolutionImage(Layer):
    def __init__(self,xShape,yShape,parameters=None,sigmaR=None):
        # first check
        if xShape[2] != yShape[2] or xShape[3] != yShape[3]:
            raise Exception('xShape {} and yShape {} should be the same in all but the first two dimensions'.format(xShape,yShape))
        if xShape[0] < yShape[0] or xShape[1] < yShape[1]:
            raise Exception('first two dimensions of xShape {} should be bigger than yShape {}'.format(xShape,yShape))
        self.xShape = xShape
        self.yShape = yShape
        
        # what size does the kernel have to be?
        # it should be xShape-yshape+1
        # if they are the same, we get 1
        # if yshape is 1 we get xShape
        self.kShape = (self.xShape[0]-self.yShape[0]+1, self.xShape[1]-self.yShape[1]+1,self.xShape[3])
        if parameters is None:
            self.parameters = np.ones(self.kShape)/np.prod(self.kShape[:2]) + np.random.randn(*self.kShape)*0.01*10.0/np.prod(self.kShape[:2])
            #for i in range(self.xShape[-1]):
            #    self.parameters[:,:,i] =  np.random.randn(*self.kShape[:-1])
            #    self.parameters[:,:,i] /= np.sum( np.power( self.parameters[:,:,i], 2) )*10.0
            #self.parameters = np.random.randn(*self.kShape)*0.01
        else:
            if parameters.shape != self.kShape:
                raise Exception('Input parameter with shape {} should have shape {}'.format(parameters.shape,self.kShape))
            self.parameters = parameters
            
        self.gradient = np.zeros(self.parameters.shape)
        self.sigmaR = sigmaR
    def applyLayer(self):
        self.y = np.zeros(self.yShape)        
        for i in xrange(self.yShape[-2]):
            for j in xrange(self.yShape[-1]):
                self.y[:,:,i,j] = sps.convolve2d(self.x[:,:,i,j],self.parameters[:,:,j],mode='valid')
        
    def applyAdjoint(self):
        # for 1d
        # say x is size N
        # w is size n
        # y is size N-n+1
        # definition of convolution
        # y[i] = sum_{j = 0}^{n-1} w[j]x[i-j]
        # this doesn't actually make sense because of boundaries
        # y[i] = sum_{j = 0}^{n-1} w[j]x[i-j+n-1]
        # now the lowest, i=0,j=n-1 gives 0-n+1+n-1=0
        # now to find adjoint write
        # sum_{i = 0}^{N-n}  y[i] [Lx][i] = sum_{i = 0}^{N-1} [L^*y][i] x[i]
        # working from the left
        # = sum_{i = 0}^{N-n}  y[i] sum_{j = 0}^{n-1} w[j] x[i-j+n-1]
        # first thing i need is to get the j out of the x
        # let i-j+n-1=k
        # so j = i-k+n-1
        # and when j = 0, k = i+n-1
        # and when j = n-1, k = i-n+1+n-1=i
        # = sum_{i = 0}^{N-n}  y[i] sum_{k = i}^{i+n-1} w[i-k+n-1] x[k]
        # = sum_{i = 0}^{N-n} sum_{k = i}^{i+n-1} y[i] w[i-k+n-1] x[k]
        # now we need to change the order of this summation, 
        # so that i's range is a function of k, and not the other way around
        # in 2d, the sum is over a stripe with slope 45 degrees, height n
        # we can take k from 0 to N-n
        # and then take i from k-n+1 to k
        # .      xoo
        # .     xxo
        # .    xxx
        # .   xxx
        # .  xxx
        # . xxx
        # .xxx
        # oxx
        #oox 
        # ..........
        # above we see a line the x axis is i from 0 to 6 (N-n=6)
        # the y axis is k, from i to i+2, i.e. n=3, i to i+n-1
        # this implies that N=9 for this exmaple
        # so we have x with size N=9 (0 to 8)
        # w with size n=3 (0 to 2)
        # y with size N-n+1=7 (0 to 6)
        # the "o"s represent continuing the pattern, these values will have to be set to zero though
        # to switch the order
        # take 
        # k from 0 to 8 (k from 0 to N-1, one for each x)
        # then i from k-2 to k (from k-n+1 to k)
        # AND
        # when we're out of bounds, we assign zero
        # = sum_{k = 0}^{N-1} sum_{i = k-n+1}^{k} y[i] w[i-k+n-1] x[k]
        # =  sum_{k = 0}^{N-1} x[k] sum_{i = k-n+1}^{k} y[i] w[i-k+n-1] 
        # do the values for w work?
        # smallest value of i-k+n-1 is k = N-1, i = k-n+1
        # then i-k+n-1 = i-N+1+n-1 = i-N+n = k-n+1-N+n = k-N+1 = N-1-N+1 = 0 (ok)
        # biggest value of i-k+n-1 is k=0, i = k
        # then i-k+n-1 = 0-0+n-1 = n-1 (ok)
        # how about for y?  I know it will go out of bounds
        # y[i] for i from k-n+1 to k
        # with k from 0 to N-1
        # so lowest value is 
        # I have to zero pad it by n-1 on BOTH sides (see diagram)
        
        ey_ = np.pad(self.ey, 
                     ((self.kShape[0]-1,self.kShape[0]-1), (self.kShape[1]-1,self.kShape[1]-1),(0,0),(0,0)), 
                     mode='constant', constant_values=(0.0,0.0) )
        self.ex = np.zeros(self.xShape)   
        for i in xrange(self.yShape[-2]):
            for j in xrange(self.yShape[-1]):
                self.ex[:,:,i,j] = sps.convolve2d(ey_[:,:,i,j], self.parameters[:,:,j], mode='valid')
        # I think I could do the same without padding by just saying mode='full'
                
    def incrementGradient(self):
        # from the definition
        # given is dL/dy
        # I need dy/dw
        # y[i] = sum_{j = 0}^{n-1} w[j]x[i-j+n-1]
        # then
        # d_de sum_{j = 0}^{n-1} (w[j]+eh[j])x[i-j+n-1] |e=0
        # = sum_{j = 0}^{n-1} h[j]x[i-j+n-1] 
        # generally the gradient is the thing I "dot" h with
        # what I actually need is dL/dy dy/dw
        # dL/dy = ey
        # so look at
        # sum_{i = 0}^{N-n} ey[i] sum_{j = 1}^{n-1} h[j]x[i-j+n-1]
        # so we see the gradient is
        # grad[j] = sum_{i = 0}^{N-n} ey[i] x[i-j+n-1]
        # this is like a convolution, but with x flipped
        # will this give the right size?
        # N-n+1 with N, gives N - (N-n+1) + 1 = n
        for i in xrange(self.xShape[-2]):
            for j in xrange(self.xShape[-1]):
                self.gradient[:,:,j] += sps.convolve2d(self.x[::-1,::-1,i,j], self.ey[:,:,i,j], mode='valid')
        
    def __repr__(self):
        return "(ConvolutoinImage Layer {0}->{1})".format(self.xShape,self.yShape)
    
'''    
#l = ConvolutionImage((64,64,3,2),(62,62,3,2),parameters=np.ones((3,3,2)))
l = ConvolutionImage((64,64,3,2),(50,50,3,2))
I = np.random.randn(64,64,3,2)
close('all')
figure()
imshow(I[:,:,0,0],interpolation='none')
colorbar()
l.x = I
l.applyLayer()
J = l.y
figure()
imshow(J[:,:,0,0],interpolation='none')
colorbar()

l.ey = J
l.applyAdjoint()
K = l.ex
figure()
imshow(K[:,:,0,0],interpolation='none')
colorbar()
'''


# for this network I want to do the following
# start with an image
# multiply
# add
# squash
# copy
#
# convolve
# downsample
# add
# sqash
# multiply
# add
# squash
# 
# repeat (the obove is like a single hidden layer)
# 
# now my images are small
# matrix
# add
# squash
# matrix
# add
# 
# a heuristic
# first time downsample to a power of 2
# repeat until I get to 1
def makeStandardConvolutionalNeuralNet(inputShape=(256,256,3,1),outputDim=1,nCopies=5,nInter=None,sigmaR=10.0):
    
    # if nInter is none we keep adding layers until we've downsampled to a size of 1
    
    # initialize
    nn = NeuralNet()
    # start by multiply shift and squash
    nn.addLayer(Multiplication(inputShape,inputShape,sigmaR=sigmaR))
    nn.addLayer(Addition(inputShape,inputShape,sigmaR=sigmaR))
    nn.addLayer(ComponentwiseFunction())
    
    # now make copies
    shape1 = (inputShape[0],inputShape[1],inputShape[2],inputShape[3]*nCopies)
    nn.addLayer(CopyImage(inputShape,shape1))
    shape0 = shape1
    
    # now downsample to a power of 2    
    pot = 2**np.floor( np.log(np.min(inputShape[:2]))/np.log(2.0))
    # find the downsample factor, convolve with a kernel of this size
    # old = new*d + r
    d0 = inputShape[0]/float(pot)
    d1 = inputShape[1]/float(pot)
    n0 = np.ceil(d0)
    n1 = np.ceil(d1)
    shape1 = (shape0[0]-n0+1,shape0[1]-n1+1,shape0[2],shape0[3])
    nn.addLayer(ConvolutionImage(shape0,shape1,sigmaR=sigmaR))
    for i in range(nCopies-1):
        nn[-1].parameters[:,:,i] -= np.mean(nn[-1].parameters[:,:,i])
        nn[-1].parameters[:,:,i] /= np.sum(np.power(nn[-1].parameters[:,:,i],2))
    shape0=shape1
    shape1=(pot,pot,shape0[2],shape0[3])
    nn.addLayer(DownsampleImage(shape0,shape1))
    shape0=shape1
    #nn.addLayer(Addition(shape0,shape1,sigmaR=sigmaR))
    #nn.addLayer(ComponentwiseFunction())
    nn.addLayer(Multiplication(shape0,shape1,sigmaR=sigmaR))
    nn.addLayer(Addition(shape0,shape1,sigmaR=sigmaR))
    nn.addLayer(ComponentwiseFunction())
     
    # now we continue this, downsampling
    while shape1[0]>1:
        shape0 = shape1
        shape1 = (shape0[0]-1,shape0[1]-1,shape0[2],shape0[3]) # a 2x2 kernel
        nn.addLayer(ConvolutionImage(shape0,shape1,sigmaR=sigmaR))        
        shape2 = (shape0[0]/2,shape0[1]/2,shape0[2],shape0[3])
        nn.addLayer(DownsampleImage(shape1,shape2))
        shape0 = shape2
        shape1 = shape2
        #nn.addLayer(Addition(shape0,shape1,sigmaR=sigmaR))
        #nn.addLayer(ComponentwiseFunction())
        nn.addLayer(Multiplication(shape0,shape1,sigmaR=sigmaR))
        nn.addLayer(Addition(shape0,shape1,sigmaR=sigmaR))
        nn.addLayer(ComponentwiseFunction())
    
    # now I'm almost done
    shape0 = shape1
    shape1 = (outputDim,1)
    nn.addLayer(Matrix(shape0,shape1,sigmaR=sigmaR))
    nn.addLayer(Addition(shape1,sigmaR=sigmaR))
    
    
    
    
    
    
    return nn
    
    
'''    
I = np.random.randn(270,250,3,1)    
I = imread('/home/dtward/Desktop/edmonton.jpg')
I = I[::2,::2,:]
I = I[::2,::2,:]
#I = I[::2,::2,:]
#I = I[::2,::2,:]
I = np.double(I)/255.0
I = np.sum(I,axis=2)/3.0
I.shape = (I.shape[0],I.shape[1],1 if len(I.shape)==2 else 3,1)
nn = makeStandardConvolutionalNeuralNet(I.shape)
J = nn(I)
close('all')
figure()
imshow(I[:,:,0,0])

for l in [l for l in nn.layers if type(l)==type(ComponentwiseFunction())]:
    figure()
    J=l.y
    n=J.shape[-1]
    for i in range(n):
        subplot(1,n,i+1)
        imshow(J[:,:,0,i])
'''

