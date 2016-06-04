# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:11:50 2016

@author: dtward
"""

# load some faces
import NN
reload(NN) # just in case I changed it
import numpy as np
import glob
import matplotlib.pyplot as plt


files = glob.glob('/home/dtward/Documents/MUCT/jpg/rigid/*.jpg')
files = files[:50]
#files = glob.glob('/home/dtward/Documents/MUCT/jpg/rigid/*.jpg')
files = glob.glob('/cis/home/dtward/Documents/illustrationsWithFaces/registered/i*.jpg')
#files = glob.glob('/cis/home/dtward/Documents/illustrationsWithFaces/registered/*qa-*.jpg')
#files = files[:10]


nSubjects = len(files)
plt.close('all')
hFig = plt.figure()
for i,f in enumerate(files):
    I = plt.imread(f)
    #I = I[225:525,125:425,:]
    I = I[250:500,150:400,:]
    I = I[::2,::2,:]
    I = I[::2,::2,:]
    #I = I[::2,::2,:]
    if i == 0:
        X = np.zeros(I.shape+(1,len(files),))
        Z = np.zeros((2,1,len(files)))
        
    hFig.clf()
    hAx = hFig.add_subplot(111)
    hAx.imshow(I,interpolation='none')
    #raise Exception
    plt.pause(0.01)
    I.shape=I.shape+(1,)
    X[:,:,:,:,i] = I.astype(np.double)/255.0
    # gender
    if f.find('-f')>0:
        Z[0,0,i] = 1.0
        Z[1,0,i] = 0.0
    else:
        Z[0,0,i] = 0.0
        Z[1,0,i] = 1.0

    
nCopies = 4
nn = NN.makeStandardConvolutionalNeuralNet(inputShape=I.shape,outputDim=2,nCopies=nCopies,sigmaR=10.0)
nn.setTrainingData(X,Z)
layersToDraw = [l for l in nn if type(l) == type(NN.ComponentwiseFunction())]
nPlots = len(layersToDraw)
hFig.clf()

hFig2 = plt.figure()


nIter = 10000
nPerDraw = 1
nRepeats = nIter/nPerDraw
epsilon = 0.0001
epsilon = 0.00001
L = 1.0e10
for repeat in range(nRepeats):    
    print('repeat {} of {}'.format(repeat,nRepeats))
    LOld = float(L)
    nn.train(nPerDraw,epsilon)
    L = float(nn.L)
    if LOld < L:
        #epsilon *= 0.9
        #print 'reducing epsilon to {}'.format(epsilon)
        pass
    else:
        #epsilon *= 1.01
        #print 'increasing epsilon to {}'.format(epsilon)
        pass
        
    randPerson = np.random.randint(nSubjects)
    nn(X[:,:,:,:,randPerson])
    hFig.clf()
    layersToDraw = [l for l in nn if type(l) == type(NN.ComponentwiseFunction())]
    nToShow = 6
    for i in range(nToShow):
        skip = 2**i
        for j in range(nCopies):
            try:
                hAx = hFig.add_subplot(nCopies,nToShow,i + j*nToShow + 1)
                Ishow = layersToDraw[i].y[:,:,:,j*skip]
                Ishow = (Ishow - np.min(Ishow))/(np.max(Ishow)-np.min(Ishow))
                hIm = hAx.imshow(Ishow,interpolation='none')
                #plt.colorbar(hIm)

            except:
                pass

    plt.pause(0.01)

    
    # now I want to take this person and make them like their opposite label
    if nn.E < len(files)*0.25 and nn.E < 1.0 and not repeat%10:
        my_input = np.array(X[:,:,:,:,randPerson])
        for i in range(100):
            reg = 1000.0
            
            hFig2.clf()
            hAx = hFig2.add_subplot(2,2,1)
            Ishow = X[:,:,:,0,randPerson]
            Ishow = (Ishow - np.min(Ishow))/(np.max(Ishow)-np.min(Ishow))
            hAx.imshow(Ishow,interpolation='none')
            target = 1.0 - Z[:,:,randPerson]
            
            
            # feed forward
            out = nn(my_input)
            print(out)
            # feed back
            e = out - target
            nn.backpropogate(e)
            # get gradient
            grad = nn[0].ex + (my_input - np.array(X[:,:,:,:,randPerson]))/reg**2
            Ishow = grad[:,:,:,0]
            Ishow = (Ishow - np.min(Ishow))/(np.max(Ishow)-np.min(Ishow))
        
            hAx = hFig2.add_subplot(2,2,2)
            hIm = hAx.imshow(Ishow,interpolation='none')
            plt.colorbar(hIm)

            
            step = 0.01

            my_input = my_input - step*grad
            Ishow = my_input[:,:,:,0]
            Ishow = (Ishow - np.min(Ishow))/(np.max(Ishow)-np.min(Ishow))
            hAx = hFig2.add_subplot(2,2,3)
            hIm = hAx.imshow(Ishow,interpolation='none')


            hAx = hFig2.add_subplot(2,2,4)
            Ishow = my_input[:,:,:,0] - X[:,:,:,0,randPerson]
            Ishow = (Ishow - np.min(Ishow))/(np.max(Ishow)-np.min(Ishow))
            hIm = hAx.imshow(Ishow,interpolation='none')
            plt.colorbar(hIm)


            E = np.sum(e**2)/2.0 + np.sum((my_input - np.array(X[:,:,:,:,randPerson]))**2)/2.0/reg**2
            
            print(E)

            plt.pause(0.01)
        
        

        
        
    
    







