#!/usr/bin/env python

import uproot
import numpy as np
from root_numpy import array2root
import os
from argparse import ArgumentParser
from multiprocessing import Pool


def tonumpy(inarr):
    return np.array(list(inarr), dtype='float32')


#makes 
def readPU(minbias_files, nevents=50, nfiles=5, nPU=200):
    select = np.array(range(len(minbias_files)))
    np.random.shuffle(select)
    select = select[:nfiles]
    #print(select)
    #open them
    #take nPU random events
    inarrs=[]
    for i in range(nfiles):
        file = minbias_files[select[i]]
        tree = uproot.open(file)["B4"]
        arr = tonumpy(tree["rechit_energy"].array() )
        arr = np.expand_dims(arr, axis=0)# 1 x nev x rh
        inarrs.append(arr)
    
    allarr = np.concatenate(inarrs, axis=0) #nfiles x nev x rh
    allarr = np.reshape(allarr, [allarr.shape[0]*allarr.shape[1],allarr.shape[2]])
    
    #print(allarr.shape)
    evtarrs=[]
    for ev in range(nevents):
        idx = np.random.randint(allarr.shape[0], size=nPU)
        evt = allarr[idx]
        evt = np.sum(evt,axis=0, keepdims=True)
        #print(evt.shape)
        evtarrs.append(evt)
    
    return np.concatenate(evtarrs,axis=0)
        
    
        

parser = ArgumentParser('premix PU')
parser.add_argument('inputFile')
parser.add_argument('-o', help='outputDir', default='')

args = parser.parse_args()

from DeepJetCore.TrainData import TrainData #heavy import here


outdir=args.o
if len(outdir) < 1:
    exit()

os.system('mkdir -p '+outdir)


allfiles = []
print("checking files")
with open(args.inputFile) as f:
    for l in f:
        l = l.rstrip('\n').rstrip(' ')
        if len(l) and os.path.isfile(l):
            try:
                tree = uproot.open(l)["B4"]
                nevents = tree.numentries
                allfiles.append(l)
            except:
                pass
            
            
            
# 100 PU per file 100 events per file 
# this reduces the effective stat by a factor of 8
def premixfile(f): 
    eventsperround=200
    neventstotal=400
    nPUpremix = 25
    nfilespremix = 5
    
    nevents=0 
    filearr=[]
    print(f)
    while nevents<100:
        print('..adding files..')          
        filearr.append(readPU(allfiles, nevents=eventsperround, nfiles=nfilespremix, nPU=nPUpremix))
        nevents+=eventsperround
    filearr = np.concatenate(filearr,axis=0)
    mean = np.mean(filearr)
    print('mean energy ',mean)
    td = TrainData()
    td._store([filearr],[],[])
    print('..writing '+os.path.basename(f)) 
    td.writeToFile(outdir+'/'+os.path.basename(f)+'_mix.djctd')
    del td




print("mixing")
#premixfile(allfiles[0])
#exit()
p = Pool(20)
p.map(premixfile, allfiles)




