#!/usr/bin/env python



from argparse import ArgumentParser
parser = ArgumentParser('makes the plots')
parser.add_argument('inputFile')
parser.add_argument('outputDir')
args=parser.parse_args()


from DeepJetCore.evaluation import makeROCs_async
import os

os.system('mkdir -p '+args.outputDir)

inst_lumi = 34
cross_section = 70

normalisation = 10. ##inst_lumi*cross_section



makeROCs_async(intextfile=args.inputFile, 
               name_list=['inclusive','E>100','||>0.5'], 
               probabilities_list='prob_isSignal', 
               truths_list='isSignal',
               vetos_list='!isSignal',
                    colors_list='auto', 
                    outpdffile=args.outputDir+'/inclusive_roc.pdf', 
                    #background has true_energy==0 and cuts are also applied to background 
                    cuts=['','(true_energy<50 || true_energy<1)','(true_howparallel>0.5 || true_energy<1)'], 
                    cmsstyle=False, 
                    firstcomment='',
                    secondcomment='',
                    invalidlist='',
                    extralegend=None, #['solid?udsg','hatched?c'])
                    logY=True,
                    individual=False,
                    xaxis="Signal efficiency",
                    yaxis="Rate [kHz]",
                    nbins=200,
                    treename='tree',
                    yscales=normalisation,
                    no_friend_tree=True)