

from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np
import uproot
import glob
import random
import matplotlib.cm as cm
from mixing import premixfile




class TrainData_calo(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        
        self.nPU=200
        
        
    #def createWeighterObjects(self, allsourcefiles):
        # 
        # return {}
    
    def tonumpy(self,inarr):
        return np.array(list(inarr), dtype='float32')
    
    def fileIsValid(self, filename):
        try:
            tree = uproot.open(filename)["B4"]
            nevents = tree.numentries
        except:
            return False
        return True
    
    def addPU(self, energy, nPU, inputfilenumber, istestsample):
        
        if nPU<1:
            return energy
        files = TrainData_calo_pufiles_train
        
        arr = premixfile(inputfilenumber,files,energy.shape[0],nPU,nfilespremix=5,eventsperround=100)

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
        
        fileTimeOut(filename, 10)#10 seconds for eos to recover 
        
        tree = uproot.open(filename)["B4"]
        nevents = tree.numentries
        
        rechit_energy = self.tonumpy(tree["rechit_energy"].array() )
        print(rechit_energy.shape)
        
        #create minbias and signal  B x V
        no_signal = np.zeros_like(rechit_energy, dtype='float32')
        
        rechit_energy = np.concatenate([rechit_energy, no_signal],axis=-1) # B x 2*V
        rechit_energy = np.reshape(rechit_energy, [nevents*2, -1]) # 2*B x V
        
        issignal = np.array([1.,0.], dtype='float32')
        issignal = np.reshape(issignal, [2,1])
        issignal = np.tile(issignal, [nevents,1])
        
        print(np.sum(rechit_energy,axis=-1))
        
        #for failed events
        issignal[np.sum(rechit_energy,axis=-1)==0] = 0 
        
        #print('issignal',issignal)
        #print('rechit_energy',rechit_energy.shape)
        
        filenumber = filename.split('_tmp_')[1]
        filenumber = int(filenumber[:-5])
        
        if readPU:
            print('adding PU')
            rechit_energy = self.addPU(rechit_energy , self.nPU, filenumber ,not istraining)
            print('PU done')
        rechit_energy = np.reshape(rechit_energy, [-1, nofEELayers, etasegments, Ncalowedges]) #nofEELayers, etasegments, Ncalowedges
        print(rechit_energy.shape)
        rechit_energy = rechit_energy.transpose((0,2,3,1))
        
        #exit()
        
        #maybe use layer number as colour modifier here?
        
        
        
        norm = np.array([[[[
            0.00010236,
            0.0002501 ,
            0.00041541,
            0.00054669,
            0.00065345,
            0.00072501,
            0.00074395,
            0.00071629,
            0.00067139,
            0.00060171,
            0.00052976,
            0.00047216,
            0.00039632,
            0.00024775,
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

        
        
        return rechit_energy, issignal
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        
        all, issignal = self.read_and_add_minbias(filename, weighterobjects, istraining, True)#istraining,True)
        
        #eal with phi modulo. it is 120/2pi, so 0.4 in phi should be enough, so 8 extra repitions
        all = np.concatenate([all, all[:,:,:8,:]],axis=2)
        
        debug=False
        if debug:
            #event=1
            for event in range(50):
                evtsforplot = np.sum(self.to_color(all[event:event+1]), axis=0)
                import matplotlib.pyplot as plt
                fig,ax =  plt.subplots(1,1)
                maxcol = np.max(evtsforplot)
                ax.imshow(evtsforplot/(maxcol+0.0001))
                #print('max energy '+str(maxcol))
                fig.savefig("event_displ"+str(event)+".pdf")
                plt.close()
        
        return [all],[issignal],[]
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        tree = uproot.open(inputfile)["B4"]
        true_energy = np.expand_dims(self.tonumpy(tree["true_energy"].array() ),axis=1)
        true_howparallel = np.expand_dims(self.tonumpy(tree["true_howparallel"].array() ),axis=1)
        
        print('true_energy' ,true_energy.shape)
        
        zeros = np.zeros_like(true_energy)
        
        true_energy = np.concatenate([true_energy,zeros],axis=-1)
        true_energy = np.reshape(true_energy, [zeros.shape[0]*2,1])
        
        true_howparallel= np.concatenate([true_howparallel,zeros],axis=-1)
        true_howparallel = np.reshape(true_howparallel, [zeros.shape[0]*2,1])
        
        print(predicted[0].shape, truth[0].shape, true_howparallel.shape, true_energy.shape)
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.concatenate([predicted[0], truth[0], true_howparallel, true_energy],axis=-1).transpose(), 
                                             names='prob_isSignal, isSignal, true_howparallel, true_energy')
        
        array2root(out, outfilename, 'tree')
        
        





class TrainData_calo_noPU(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        
        self.nPU=0


def readFileList(path):
    import os
    files=[]
    with open(path) as f:
        for l in f:
            l = l.rstrip('\n').rstrip(' ')
            if len(l) and os.path.isfile(l):
                files.append(l)
    return files
    
    

TrainData_calo_pufiles_train=readFileList("/eos/home-j/jkiesele/DeepNtuples/testminbias/good_files.txt")

