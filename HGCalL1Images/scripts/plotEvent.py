#!/usr/bin/env python

import uproot
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from argparse import ArgumentParser


parser = ArgumentParser('make pretty plot')
parser.add_argument('inputFile')

args = parser.parse_args()


def readFile(filename):
    
    def tonumpy(inarr):
        return np.array(list(inarr), dtype='float32')

    out={}
    tree = uproot.open(filename)["B4"]
    nevents = tree.numentries    
    out['rechit_energy'] = tonumpy(tree["rechit_energy"].array() )
    out['rechit_eta'] = tonumpy(tree["rechit_eta"].array() )
    out['rechit_phi'] = tonumpy(tree["rechit_phi"].array() )
    out['rechit_layer'] = tonumpy(tree["rechit_layer"].array() )
    out['true_angle'] = tonumpy(tree["true_angle"].array() )
    out['true_energy'] = tonumpy(tree["true_energy"].array() )

    return out



def to_color(rechit_image):
    colors = np.array(range(14),dtype='float')/14. #layer colours
    colors = cm.rainbow(colors)
    colors = colors[:,:-1]
    colors = np.reshape(colors, [14,1,1,3])
    rechit_energy = np.expand_dims(rechit_image, axis=-1)
    rechit_energy = np.array(rechit_energy*colors, dtype='float32')
    return np.sum(rechit_energy, axis=0)

def addScatter(datadict, event, ax):
    
    #reshape, project etc
    
    nofEELayers = 14;
    Ncalowedges=120;
    etasegments=30;
    norm = 1326.9427*1e-5
    
    
    rechit_energy = np.reshape(d['rechit_energy'][event], [ nofEELayers, etasegments, Ncalowedges])/norm
    print('rechit_energymean ',np.mean(d['rechit_energy']))
    # sum for energy, mean for coordinates
    # use layer number for colour
    colour = to_color(np.log(rechit_energy+1))
    
    rechit_energy = np.sum(rechit_energy,axis=0)
    rechit_energy = np.reshape(rechit_energy, [-1])
    
    print('rechit_energy', rechit_energy.shape)
    
    print('colour', colour.shape)
    eta = np.reshape(d['rechit_eta'][event], [ nofEELayers, etasegments, Ncalowedges])
    phi = np.reshape(d['rechit_phi'][event], [ nofEELayers, etasegments, Ncalowedges])
    
    eta = np.reshape(np.mean(eta, axis=0), [-1])
    phi = np.reshape(np.mean(phi, axis=0), [-1])
    colour = np.reshape(colour, [-1,3])/np.max(colour)
    
    print('maxcol',np.max(colour))
    
    
    #colour = np.concatenate([colour, np.zeros_like(colour[:,0:1])+1],axis=-1)
    #colour = mpl.colors.to_hex(colour)
    print('colour',colour.shape)
    #
    print('true energy',event, d['true_energy'][event])
    
    #print(colour)
    
    print(event, d['true_angle'][event])
    
    size_scaling = mpl.rcParams['lines.markersize']**2 * np.log(rechit_energy+1) #(0.01+ np.log(rechit_energy*5+1))/4.
    
    
    sorting = np.reshape(np.argsort(rechit_energy[rechit_energy>0], axis=0), [-1])
    
    
    ax.scatter(phi[rechit_energy>0][sorting],
                  eta[rechit_energy>0][sorting],
                  c=colour[rechit_energy>0][sorting],
                  s=size_scaling[rechit_energy>0][sorting])
    return ax

labelsize=16
axtitlesize=16

d = readFile(args.inputFile)

params = {'axes.labelsize': labelsize,
          'axes.titlesize': axtitlesize}

allevs = np.array(range(800))

true_angle = d['true_angle']
for t in [0.01,0.05,0.1]:
    sel = true_angle<t
    if np.any(sel) :
        print(t)
        print(allevs[sel])


#for event in [18,366]:

plt.rcParams.update(params)

fig,ax =  plt.subplots(1,1)

ax=addScatter(d, 18, ax)
ax=addScatter(d, 366, ax)
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\eta$')

ax.set_aspect(1.7)

ax.tick_params(axis='y', labelsize=labelsize)
ax.tick_params(axis='x', labelsize=labelsize)
plt.tight_layout(pad=0.15)
fig.savefig("plot_projection_white.pdf")
ax.set_facecolor('black')
fig.savefig("plot_projection.pdf")
plt.close()












