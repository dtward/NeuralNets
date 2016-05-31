# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:11:50 2016

@author: dtward
"""

# load some faces
import NN
import numpy as np
import glob
import matplotlib.pyplot as plt
files = glob.glob('/home/dtward/Documents/MUCT/jpg/rigid/*.jpg')
#files = files[:50]
nSubjects = len(files)
plt.close('all')
hFig = plt.figure()
for i,f in enumerate(files):
    I = plt.imread(f)
    #I = I[225:525,125:425,:]
    I = I[250:500,150:400,:]
    I = I[::2,::2,:]
    #I = I[::2,::2,:]
    #I = I[::2,::2,:]
    if i == 0:
        X = np.zeros(I.shape+(1,len(files),))
        Z = np.zeros((2,1,len(files)))
        
    hFig.clf()
    hAx = hFig.add_subplot(111)
    hAx.imshow(I)
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

    
nCopies = 3
nn = NN.makeStandardConvolutionalNeuralNet(inputShape=I.shape,outputDim=2,nCopies=nCopies,sigmaR=100.0)
nn.setTrainingData(X,Z)
layersToDraw = [l for l in nn if type(l) == type(NN.ComponentwiseFunction())]
nPlots = len(layersToDraw)
hFig.clf()
nIter = 10000
nPerDraw = 1
nRepeats = nIter/nPerDraw
for repeat in range(nRepeats):    
    print('repeat {} of {}'.format(repeat,nRepeats))
    nn.train(nPerDraw,0.001)
    randPerson = np.random.randint(nSubjects)
    nn(X[:,:,:,:,randPerson])
    hFig.clf()
    layersToDraw = [l for l in nn if type(l) == type(NN.ComponentwiseFunction())]
    nToShow = 5
    for i in range(nToShow):              
        hAx = hFig.add_subplot(nCopies,nToShow,i+1)
        Ishow = layersToDraw[i].y[:,:,:,0]
        Ishow = (Ishow - np.min(Ishow))/(np.max(Ishow)-np.min(Ishow))
        hIm = hAx.imshow(Ishow,interpolation='none')
        #plt.colorbar(hIm)
        try:
            hAx = hFig.add_subplot(nCopies,nToShow,i+nToShow+1)
            Ishow = layersToDraw[i].y[:,:,:,1]
            Ishow = (Ishow - np.min(Ishow))/(np.max(Ishow)-np.min(Ishow))
            hIm = hAx.imshow(Ishow,interpolation='none')
            #plt.colorbar(hIm)
        except:
            pass
        try:
            hAx = hFig.add_subplot(nCopies,nToShow,i+nToShow*2+1)
            Ishow = layersToDraw[i].y[:,:,:,2]
            Ishow = (Ishow - np.min(Ishow))/(np.max(Ishow)-np.min(Ishow))
            hIm = hAx.imshow(Ishow,interpolation='none')
            #plt.colorbar(hIm)
        except:
            pass
    plt.pause(0.1)
        
        
    
    







