#!/usr/bin/env python3



from argparse import ArgumentParser
parser = ArgumentParser('makes the plots')
parser.add_argument('inputFile')
parser.add_argument('outputDir')
args=parser.parse_args()


from DeepJetCore.evaluation import makeROCs_async, makeEffPlots_async
import os

os.system('mkdir -p '+args.outputDir)

inst_lumi = 34
cross_section = 70

normalisation = 2760*11.246 ##inst_lumi*cross_section

makeEffPlots_async(intextfile=args.inputFile, 
                   name_list=['#alpha = [0.0,0.2]',
                          '#alpha = [0.2,0.4]',
                          '#alpha = [0.4,0.6]',
                          '#alpha = [0.6, inf]'
                          ], 
                   variables='true_energy', 
                   cutsnum=['isSignal && prob_isSignal>0.991582 && (true_angle>0.0 && true_angle<=0.2) ',
                            'isSignal && prob_isSignal>0.991582 && (true_angle>0.2 && true_angle<=0.4) ',
                            'isSignal && prob_isSignal>0.991582 && (true_angle>0.4 && true_angle<=0.6) ',
                            'isSignal && prob_isSignal>0.991582 && (true_angle>0.6 && true_angle<=100) ',
                           ],
                   cutsden=['isSignal && (true_angle>0.0 && true_angle<=0.2) ', 
                            'isSignal && (true_angle>0.2 && true_angle<=0.3) ', 
                            'isSignal && (true_angle>0.4 && true_angle<=0.6) ', 
                            'isSignal && (true_angle>0.6 && true_angle<=100) ', 
                            ],
                   colours='auto',
                   outpdffile=args.outputDir+'/efficiency_at_15kHz_energy.pdf', 
                   xaxis='Photon energy [GeV]',
                   yaxis='Efficiency',
                   minimum=1e100,maximum=-1e100,
                   nbins=9, SetLogY = False, Xmin = 10, Xmax = 190 ,
                   treename="tree")

makeEffPlots_async(intextfile=args.inputFile, 
                   name_list=['E = [10, 30] GeV',
                          'E = [30, 50] GeV',
                          'E = [50, 70] GeV',
                          'E = [70, 200] GeV'
                          ], 
                   variables='true_angle', 
                   cutsnum=['isSignal && prob_isSignal>0.991582 && (true_energy>10  && true_energy<=30) ',
                            'isSignal && prob_isSignal>0.991582 && (true_energy>30  && true_energy<=50) ',
                            'isSignal && prob_isSignal>0.991582 && (true_energy>50  && true_energy<=70) ',
                            'isSignal && prob_isSignal>0.991582 && (true_energy>70  && true_energy<=200) ',
                           ],
                   cutsden=['isSignal && (true_energy>10  && true_energy<=30) ', 
                            'isSignal && (true_energy>30  && true_energy<=50) ', 
                            'isSignal && (true_energy>50  && true_energy<=70) ', 
                            'isSignal && (true_energy>70  && true_energy<=200)', 
                            ],
                   colours='auto',
                   outpdffile=args.outputDir+'/efficiency_at_15kHz_alpha.pdf', 
                   xaxis='#alpha',
                   yaxis='Efficiency',
                   minimum=1e100,maximum=-1e100,
                   nbins=10, SetLogY = False, Xmin = 0.0, Xmax = 1.0471975512 ,
                   treename="tree")


makeROCs_async(intextfile=args.inputFile, 
               name_list=['E = [10, 30] GeV',
                          'E = [30, 50] GeV',
                          'E = [50, 70] GeV',
                          'E = [70, 200] GeV'], 
               probabilities_list='prob_isSignal', 
               truths_list='isSignal',
               vetos_list='!isSignal',
                    colors_list='auto', 
                    outpdffile=args.outputDir+'/energy_roc.pdf', 
                    #background has true_energy==0 and cuts are also applied to background 
                    cuts=['((true_energy>10 && true_energy<=30) || !isSignal)',
                          '((true_energy>30 && true_energy<=50) || !isSignal)',
                          '((true_energy>50 && true_energy<=70) || !isSignal)',
                          '((true_energy>70 && true_energy<=200) || !isSignal)'], 
                    cmsstyle=False, 
                    firstcomment='',
                    secondcomment='',
                    invalidlist='',
                    extralegend=None, #['solid?udsg','hatched?c'])
                    logY=True,
                    individual=False,
                    xaxis="Signal efficiency",
                    yaxis="Rate [kHz]",
                    nbins=500,
                    treename='tree',
                    yscales=normalisation,
                    no_friend_tree=True)

exit()

makeROCs_async(intextfile=args.inputFile, 
               name_list=['inclusive','E>100','#alpha>0.3'], 
               probabilities_list='prob_isSignal', 
               truths_list='isSignal',
               vetos_list='!isSignal',
                    colors_list='auto', 
                    outpdffile=args.outputDir+'/inclusive_roc.pdf', 
                    #background has true_energy==0 and cuts are also applied to background 
                    cuts=['','(true_energy>100 || !isSignal)','(true_angle>0.3 || !isSignal)'], 
                    cmsstyle=False, 
                    firstcomment='',
                    secondcomment='',
                    invalidlist='',
                    extralegend=None, #['solid?udsg','hatched?c'])
                    logY=True,
                    individual=False,
                    xaxis="Signal efficiency",
                    yaxis="Rate [kHz]",
                    nbins=500,
                    treename='tree',
                    yscales=normalisation,
                    no_friend_tree=True)


#exit()

makeROCs_async(intextfile=args.inputFile, 
               name_list=['#alpha = [0.0,0.1]',
                          '#alpha = [0.1,0.2]',
                          '#alpha = [0.2,0.3]',
                          '#alpha = [0.3,0.6]',
                          '#alpha = [0.6, inf]',], 
               probabilities_list='prob_isSignal', 
               truths_list='isSignal',
               vetos_list='!isSignal',
                    colors_list='auto', 
                    outpdffile=args.outputDir+'/alpha_roc.pdf', 
                    #background has true_energy==0 and cuts are also applied to background 
                    cuts=['((true_angle>0.0 && true_angle<=0.1) || !isSignal)',
                          '((true_angle>0.1 && true_angle<=0.2) || !isSignal)',
                          '((true_angle>0.2 && true_angle<=0.3) || !isSignal)',
                          '((true_angle>0.3 && true_angle<=0.6) || !isSignal)',
                          '((true_angle>0.6 && true_angle<=100.) || !isSignal)'], 
                    cmsstyle=False, 
                    firstcomment='',
                    secondcomment='',
                    invalidlist='',
                    extralegend=None, #['solid?udsg','hatched?c'])
                    logY=True,
                    individual=False,
                    xaxis="Signal efficiency",
                    yaxis="Rate [kHz]",
                    nbins=500,
                    treename='tree',
                    yscales=normalisation,
                    no_friend_tree=True)



makeROCs_async(intextfile=args.inputFile, 
               name_list=['#alpha = [0.0,0.1]',
                          '#alpha = [0.1,0.2]',
                          '#alpha = [0.2,0.3]',
                          '#alpha = [0.3,0.6]',
                          '#alpha = [0.6, inf]',], 
               probabilities_list='prob_isSignal', 
               truths_list='isSignal',
               vetos_list='!isSignal',
                    colors_list='auto', 
                    outpdffile=args.outputDir+'/alpha_roc_E_20_40.pdf', 
                    #background has true_energy==0 and cuts are also applied to background 
                    cuts=['((true_energy>20 && true_energy<=40 && true_angle>0.0 && true_angle<=0.1) || !isSignal)',
                          '((true_energy>20 && true_energy<=40 && true_angle>0.1 && true_angle<=0.2) || !isSignal)',
                          '((true_energy>20 && true_energy<=40 && true_angle>0.2 && true_angle<=0.3) || !isSignal)',
                          '((true_energy>20 && true_energy<=40 && true_angle>0.3 && true_angle<=0.6) || !isSignal)',
                          '((true_energy>20 && true_energy<=40 && true_angle>0.6 && true_angle<=100.) || !isSignal)'], 
                    cmsstyle=False, 
                    firstcomment='',
                    secondcomment='',
                    invalidlist='',
                    extralegend=None, #['solid?udsg','hatched?c'])
                    logY=True,
                    individual=False,
                    xaxis="Signal efficiency",
                    yaxis="Rate [kHz]",
                    nbins=500,
                    treename='tree',
                    yscales=normalisation,
                    no_friend_tree=True)

makeROCs_async(intextfile=args.inputFile, 
               name_list=['#alpha = [0.0,0.1]',
                          '#alpha = [0.1,0.2]',
                          '#alpha = [0.2,0.3]',
                          '#alpha = [0.3,0.6]',
                          '#alpha = [0.6, inf]',], 
               probabilities_list='prob_isSignal', 
               truths_list='isSignal',
               vetos_list='!isSignal',
                    colors_list='auto', 
                    outpdffile=args.outputDir+'/alpha_roc_E_40_60.pdf', 
                    #background has true_energy==0 and cuts are also applied to background 
                    cuts=['((true_energy>40 && true_energy<=60 && true_angle>0.0 && true_angle<=0.1) || !isSignal)',
                          '((true_energy>40 && true_energy<=60 && true_angle>0.1 && true_angle<=0.2) || !isSignal)',
                          '((true_energy>40 && true_energy<=60 && true_angle>0.2 && true_angle<=0.3) || !isSignal)',
                          '((true_energy>40 && true_energy<=60 && true_angle>0.3 && true_angle<=0.6) || !isSignal)',
                          '((true_energy>40 && true_energy<=60 && true_angle>0.6 && true_angle<=100.) || !isSignal)'], 
                    cmsstyle=False, 
                    firstcomment='',
                    secondcomment='',
                    invalidlist='',
                    extralegend=None, #['solid?udsg','hatched?c'])
                    logY=True,
                    individual=False,
                    xaxis="Signal efficiency",
                    yaxis="Rate [kHz]",
                    nbins=500,
                    treename='tree',
                    yscales=normalisation,
                    no_friend_tree=True)





makeROCs_async(intextfile=args.inputFile, 
               name_list=['E = [10, 20] GeV',
                          'E = [20, 30] GeV',
                          'E = [30, 40] GeV',
                          'E = [40, 60] GeV',
                          'E = [60, 200] GeV'], 
               probabilities_list='prob_isSignal', 
               truths_list='isSignal',
               vetos_list='!isSignal',
                    colors_list='auto', 
                    outpdffile=args.outputDir+'/energy_roc_alpha0.0_0.2.pdf', 
                    #background has true_energy==0 and cuts are also applied to background 
                    cuts=['((true_angle<0.2 && true_energy>10 && true_energy<=20) || !isSignal)',
                          '((true_angle<0.2 && true_energy>20 && true_energy<=30) || !isSignal)',
                          '((true_angle<0.2 && true_energy>30 && true_energy<=40) || !isSignal)',
                          '((true_angle<0.2 && true_energy>40 && true_energy<=60) || !isSignal)',
                          '((true_angle<0.2 && true_energy>60 && true_energy<=200) || !isSignal)'], 
                    cmsstyle=False, 
                    firstcomment='',
                    secondcomment='',
                    invalidlist='',
                    extralegend=None, #['solid?udsg','hatched?c'])
                    logY=True,
                    individual=False,
                    xaxis="Signal efficiency",
                    yaxis="Rate [kHz]",
                    nbins=500,
                    treename='tree',
                    yscales=normalisation,
                    no_friend_tree=True)

makeROCs_async(intextfile=args.inputFile, 
               name_list=['E = [10, 20] GeV',
                          'E = [20, 30] GeV',
                          'E = [30, 40] GeV',
                          'E = [40, 60] GeV',
                          'E = [60, 200] GeV'], 
               probabilities_list='prob_isSignal', 
               truths_list='isSignal',
               vetos_list='!isSignal',
                    colors_list='auto', 
                    outpdffile=args.outputDir+'/energy_roc_alpha0.2_0.4.pdf', 
                    #background has true_energy==0 and cuts are also applied to background 
                    cuts=['((true_angle>0.2 && true_angle<0.4 && true_energy>10 && true_energy<=20) || !isSignal)',
                          '((true_angle>0.2 && true_angle<0.4 && true_energy>20 && true_energy<=30) || !isSignal)',
                          '((true_angle>0.2 && true_angle<0.4 && true_energy>30 && true_energy<=40) || !isSignal)',
                          '((true_angle>0.2 && true_angle<0.4 && true_energy>40 && true_energy<=60) || !isSignal)',
                          '((true_angle>0.2 && true_angle<0.4 && true_energy>60 && true_energy<=200) || !isSignal)'], 
                    cmsstyle=False, 
                    firstcomment='',
                    secondcomment='',
                    invalidlist='',
                    extralegend=None, #['solid?udsg','hatched?c'])
                    logY=True,
                    individual=False,
                    xaxis="Signal efficiency",
                    yaxis="Rate [kHz]",
                    nbins=500,
                    treename='tree',
                    yscales=normalisation,
                    no_friend_tree=True)


makeROCs_async(intextfile=args.inputFile, 
               name_list=['E = [10, 20] GeV',
                          'E = [20, 30] GeV',
                          'E = [30, 40] GeV',
                          'E = [40, 60] GeV',
                          'E = [60, 200] GeV'], 
               probabilities_list='prob_isSignal', 
               truths_list='isSignal',
               vetos_list='!isSignal',
                    colors_list='auto', 
                    outpdffile=args.outputDir+'/energy_roc_alpha0.4_0.6.pdf', 
                    #background has true_energy==0 and cuts are also applied to background 
                    cuts=['((true_angle>0.4 && true_angle<0.6 && true_energy>10 && true_energy<=20) || !isSignal)',
                          '((true_angle>0.4 && true_angle<0.6 && true_energy>20 && true_energy<=30) || !isSignal)',
                          '((true_angle>0.4 && true_angle<0.6 && true_energy>30 && true_energy<=40) || !isSignal)',
                          '((true_angle>0.4 && true_angle<0.6 && true_energy>40 && true_energy<=60) || !isSignal)',
                          '((true_angle>0.4 && true_angle<0.6 && true_energy>60 && true_energy<=200) || !isSignal)'], 
                    cmsstyle=False, 
                    firstcomment='',
                    secondcomment='',
                    invalidlist='',
                    extralegend=None, #['solid?udsg','hatched?c'])
                    logY=True,
                    individual=False,
                    xaxis="Signal efficiency",
                    yaxis="Rate [kHz]",
                    nbins=500,
                    treename='tree',
                    yscales=normalisation,
                    no_friend_tree=True)



makeROCs_async(intextfile=args.inputFile, 
               name_list=['E = [10, 20] GeV',
                          'E = [20, 30] GeV',
                          'E = [30, 40] GeV',
                          'E = [40, 60] GeV',
                          'E = [60, 200] GeV'], 
               probabilities_list='prob_isSignal', 
               truths_list='isSignal',
               vetos_list='!isSignal',
                    colors_list='auto', 
                    outpdffile=args.outputDir+'/energy_roc_alpha0.6.pdf', 
                    #background has true_energy==0 and cuts are also applied to background 
                    cuts=['((true_angle>0.6 && true_energy>10 && true_energy<=20) || !isSignal)',
                          '((true_angle>0.6 && true_energy>20 && true_energy<=30) || !isSignal)',
                          '((true_angle>0.6 && true_energy>30 && true_energy<=40) || !isSignal)',
                          '((true_angle>0.6 && true_energy>40 && true_energy<=60) || !isSignal)',
                          '((true_angle>0.6 && true_energy>60 && true_energy<=200) || !isSignal)'], 
                    cmsstyle=False, 
                    firstcomment='',
                    secondcomment='',
                    invalidlist='',
                    extralegend=None, #['solid?udsg','hatched?c'])
                    logY=True,
                    individual=False,
                    xaxis="Signal efficiency",
                    yaxis="Rate [kHz]",
                    nbins=500,
                    treename='tree',
                    yscales=normalisation,
                    no_friend_tree=True)




## some efficiency curves at 15kHz rate (train2: prob >= 0.991582)










