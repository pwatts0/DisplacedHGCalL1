#!/usr/bin/env python3

import uproot
import numpy as np
from root_numpy import array2root
import os
from argparse import ArgumentParser
from multiprocessing import Pool


def tonumpy(inarr):
    return np.array(list(inarr), dtype='float32')

parser = ArgumentParser('Skim to rechit energy only')
parser.add_argument('inputFile')
parser.add_argument('-o', help='outputDir', default='')

args = parser.parse_args()

outdir=args.o
if len(outdir) < 1:
    exit()

os.system('mkdir -p '+outdir)

def convertAndWrite(infile):
    
    thisofile=outdir+'/'+os.path.basename(infile)[:-5]+'_skim.root'
    os.system('skim '+infile+' '+thisofile)
     

allfiles = []
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
        
print(allfiles)        
p = Pool()
p.map(convertAndWrite, allfiles)
