

from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np
import uproot
import glob
import random
import matplotlib.cm as cm
from mixing import premixfile
import os

import ROOT



def readFileList(path):
    print('verifying minbias file list for '+path)
    files=[]
    dir = os.path.dirname(path)
    with open(path) as f:
        for l in f:
            l = dir+'/'+l.rstrip('\n').rstrip(' ')
            if len(l):# # and os.path.isfile(l):
                files.append(l)
    return files


if "cmg-gpu1080" in os.getenv("HOSTNAME"):
    TrainData_calo_pufiles_test=readFileList("/home/scratch/jkiesele/minbias_test/files.txt")
    TrainData_calo_pufiles_train=readFileList("/data/hgcal-0/store/jkiesele/Displ_Calo/minbias/minbias_train/files.txt")
else:
    TrainData_calo_pufiles_test=readFileList("/eos/home-j/jalimena/DisplacedCalo/minbias_test/files.txt")
    TrainData_calo_pufiles_train=readFileList("/eos/home-j/jalimena/DisplacedCalo/minbias_train/files.txt")
    
class TrainData_calo(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        
        self.nPU=200
        self.nfilespremix=20
        self.eventsperround=200
        self.always_use_test_minbias=False
        
        
    #def createWeighterObjects(self, allsourcefiles):
        # 
        # return {}
    
    def tonumpy(self,inarr):
        return np.array(list(inarr), dtype='float32')
    
    def fileIsValid(self, filename):
        try:
            #nevents = uproot.numentries(filename, "B4")
            f=ROOT.TFile.Open(filename)
            t=f.Get("B4")
            if t.GetEntries() < 1:
                raise ValueError("")
        except:
            return False
        return True
    
    def addPU_direct(self, energy, filenumber, istestsample):
        
        files = TrainData_calo_pufiles_train_double
        files = files[filenumber]
        print('adding files', files , ' to signal ', filenumber)
        
        pu_energy=[]
        for f in files:
            tree = uproot.open(f)["B4"]
            pu_energy.append(self.tonumpy(tree["rechit_energy"].array() ))

        pu_energy = np.concatenate(pu_energy,axis=0)
        print('pu energy', np.sum(pu_energy, axis=-1))
        print('signal energy', np.sum(energy, axis=-1))
        energy+=pu_energy
        
        return energy
    
    def addPU(self, energy, nPU, istestsample, seed):
        
        if nPU<1:
            return energy
        files = TrainData_calo_pufiles_train
        
        if istestsample or self.always_use_test_minbias:
            files = TrainData_calo_pufiles_test
            print('using test data')
        
        arr = premixfile(files,energy.shape[0],nPU,nfilespremix=self.nfilespremix,
                         eventsperround=self.eventsperround, seed=seed)
        energy+=arr
        return energy
        
    def to_color(self, rechit_image):
        colors = np.array(range(14),dtype='float')/14. #layer colours
        colors = cm.rainbow(colors)
        colors = colors[:,:-1]
        colors = np.reshape(colors, [1,1,1,14,3])
        rechit_energy = np.expand_dims(rechit_image, axis=-1)
        rechit_energy = np.array(rechit_energy*colors, dtype='float32')
        return np.sum(rechit_energy, axis=3)
    
    
    def mix_signal_info(self, signal_in, zeros_per_signal):
        if zeros_per_signal < 1:
            return signal_in
        #print('signal_in',signal_in.shape)
        no_signal = np.zeros_like(signal_in, dtype='float32') # B x V
        no_signal = np.tile(no_signal,[1,zeros_per_signal])
        #print('no_signal',no_signal.shape)
        added_zeros = np.concatenate([signal_in,no_signal],axis=1) # B x (N+1)V
        return np.reshape(added_zeros, [(zeros_per_signal+1)*signal_in.shape[0], -1]) # (N+1)B x V
        
    
    def read_and_add_minbias(self, filename, weighterobjects, istraining, readPU):
        
        #project to 3 planes
        #assign 'colour' for each
        #concat (->rgb plot)
        #
        #
        nofEELayers = 14;
        Ncalowedges=120;
        # nofHB=0;
        etasegments=30;
        #
        #
        # batch, layer, eta, phi
        
        #istraining=True
        
        if not self.fileIsValid(filename):#30 seconds for eos to recover 
            raise Exception("File "+filename+" could not be read")
        
        tree = uproot.open(filename)["B4"]
        nevents = tree.numentries
        
        rechit_energy = self.tonumpy(tree["rechit_energy"].array() )
        print(rechit_energy.shape)
        
        minbias_fraction=1
        if not istraining:
            minbias_fraction=50 #100 #work with this later, TBI
            self.nfilespremix=10
            self.eventsperround=100
        
        nevents = rechit_energy.shape[0]
        rechit_energy = self.mix_signal_info(rechit_energy, minbias_fraction)
        issignal = np.expand_dims(np.zeros(nevents,dtype='float32'),axis=1)+1.
        issignal = self.mix_signal_info(issignal, minbias_fraction)
       
        #for failed events
        issignal[np.sum(rechit_energy,axis=-1)==0] = 0 
        
        
        filenumber = os.path.basename(filename)
        if "_tmp_" in filenumber:
            filenumber = filenumber.split("_tmp_")[1]
        filenumber=filenumber[:-5]
        filenumber=filenumber.split("_")
        filenumber= int(filenumber[1])+int(filenumber[2])
        seed=filenumber#remove .root
        #print('seed',seed)
        
        
        if readPU:
            print('adding PU')
            rechit_energy = self.addPU(rechit_energy , self.nPU ,not istraining, seed)
            print('PU done')
            
        #rechit_energy = self.addPU_direct(rechit_energy,filenumber,not istraining)
            
        print(np.sum(rechit_energy,axis=-1))
        
        rechit_energy = np.reshape(rechit_energy, [-1, nofEELayers, etasegments, Ncalowedges]) #nofEELayers, etasegments, Ncalowedges
        print(rechit_energy.shape)
        rechit_energy = rechit_energy.transpose((0,2,3,1))
        
        #exit()
        
        #maybe use layer number as colour modifier here?
        
        
        
        norm = np.array([[[[
         0.0002,#   0.00010236,
         0.0002,#   0.0002501 ,
         0.0002,#   0.00041541,
         0.0002,#   0.00054669,
         0.0002,#   0.00065345,
         0.0002,#   0.00072501,
         0.0002,#   0.00074395,
         0.0002,#   0.00071629,
         0.0002,#   0.00067139,
         0.0002,#   0.00060171,
         0.0002,#   0.00052976,
         0.0002,#   0.00047216,
         0.0002,#   0.00039632,
         0.0002,#   0.00024775,
            ]]]],dtype='float32')
        
        rechit_energy = np.ascontiguousarray(rechit_energy/norm,dtype='float32')
        
        print(rechit_energy.shape)
        
        ##rechit energy: B x eta x phi x layerEn (14)
        #
        #first_projection = np.sum(rechit_energy[:,:,:,0:],axis=-1,keepdims=True)/2.#/1.5816298e-05/0.65174264
        #sec_projection   = np.sum(rechit_energy[:,:,:,7:10],axis=-1,keepdims=True)#/3.0695286e-05/0.9940545
        #third_projection = np.sum(rechit_energy[:,:,:,10:],axis=-1,keepdims=True)#/1.6368242e-05/1.3476628
        #
        #all = first_projection #np.concatenate([first_projection,sec_projection,third_projection], axis=-1)
        #

        if not istraining:
            #repeat these n-nonsignal+nsignal times  # use minbiasfraction
            true_energy = self.mix_signal_info(np.expand_dims(self.tonumpy(tree["true_energy"].array() ),axis=1), minbias_fraction)
            true_howparallel = self.mix_signal_info(np.expand_dims(self.tonumpy(tree["true_angle"].array() ),axis=1), minbias_fraction)
            
            return rechit_energy, np.concatenate([issignal, true_energy, true_howparallel], axis=-1)
        
        return rechit_energy, issignal
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        
        all, issignal = self.read_and_add_minbias(filename, weighterobjects, istraining, True)#istraining,True)
        
        #eal with phi modulo. it is 120/2pi, so 0.4 in phi should be enough, so 8 extra repitions
        all = np.concatenate([all, all[:,:,:8,:]],axis=2)
        
        debug=False
        if debug:
            nevents=50
            #maxcol = np.max(np.sum(self.to_color(all[0:nevents]), axis=0)) #just use bunch for 
            for event in range(nevents):
                evtsforplot = np.sum(self.to_color(all[event:event+1]), axis=0)
                import matplotlib.pyplot as plt
                fig,ax =  plt.subplots(1,1)
                maxcol = np.max(evtsforplot)
                ax.imshow(evtsforplot/(maxcol+0.01))
                #print('max energy '+str(maxcol))
                fig.savefig("event_displ"+str(event)+".pdf")
                plt.close()
        
        
        print('writing signal ',issignal.shape)
        return [all],[issignal],[]
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        issignal, true_energy, true_howparallel = truth[0][:,0:1], truth[0][:,1:2], truth[0][:,2:3]
        out = np.concatenate([predicted[0], issignal, true_energy, true_howparallel],axis=-1)
        names = 'prob_isSignal, isSignal, true_energy, true_angle'
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(out.transpose(), 
                                             names=names)
        
        array2root(out, outfilename+".root", 'tree')
        
        


class TrainData_calo_val(TrainData_calo):
    def __init__(self):
        TrainData_calo.__init__(self)
        
        self.always_use_test_minbias=True
        


class TrainData_calo_noPU(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        
        self.nPU=0

