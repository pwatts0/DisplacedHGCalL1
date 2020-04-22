#!/usr/bin/env python3


from mixing import getAndCheck
from argparse import ArgumentParser
import os

parser = ArgumentParser('check PU files in dir and create text file with valid files')
parser.add_argument('inputDir')
parser.add_argument('outFile')
args = parser.parse_args()

print('checking... get a cup of coffee...')
all = getAndCheck(args.inputDir)

cwd = os.getcwd()

f = open(args.outFile,"w")
for l in all:
    f.write(cwd+'/'+args.inputDir+'/'+l+'\n')
f.close()    