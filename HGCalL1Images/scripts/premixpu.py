#!/usr/bin/env python3

import uproot
import numpy as np
import os
from argparse import ArgumentParser
from multiprocessing import Pool


from mixing import  tonumpy, readPU
from mixing import premixfile as pmf
        

parser = ArgumentParser('premix PU')
parser.add_argument('inputFile')
parser.add_argument('nOutput', help='number of mixed output files',type=int)
parser.add_argument('outputDir')
parser.add_argument('-e', help='number of events per output file', default=800, type=int)
parser.add_argument('--pu', help='pileup', default=200, type=int)

args = parser.parse_args()

nPU=args.pu
nEvents=args.e
outputDir=args.outputDir
nOutput=args.nOutput
if len(outputDir) < 1:
    exit()
    


os.system('mkdir -p '+outputDir)


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
                print('file '+l+ ' seems to have problems, skipping')
            
            


#bigger import 
from DeepJetCore.TrainData import TrainData #heavy import here  

def premixfile(i): 

    eventsperround=100
    neventstotal=nEvents
    nPUpremix = nPU
    nfilespremix = 5
    
    filearr = pmf(allfiles,neventstotal,nPUpremix,nfilespremix=5,eventsperround=100)
    
    print('nevents', filearr.shape[0])
    td = TrainData()
    td._store([filearr],[],[])
    print('..writing '+str(i))
     
    td.writeToFile(outputDir+'/'+str(i)+'_mix.djctd')
    del td
    



print("mixing")
#premixfile(allfiles[0])
#exit()
indices=[i for i in range(nOutput)]

p = Pool(min(20, nOutput))
p.map(premixfile, indices)




