

from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np
import uproot
import glob
import random
import matplotlib.cm as cm

class TrainData_calo(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        
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
    
    def addPU(self, energy, nevents, nPU):
        
        if nPU<1:
            return energy
        #each file has 20 PU
        pufolder="/data/hgcal-0/store/jkiesele/Displ_Calo_minbias/400ev_25PU/"
        minbias_files= glob.glob(pufolder+"*.djctd")

        select = np.array(range(len(minbias_files)))
        
        if nPU%100:
            raise ValueError("nPU must be multiples of 100")
        
        befpu_en = np.sum(energy)
        
        for i in range(nPU/25): #100PU per event
            allpu=[]
            eventshere = 0
            np.random.shuffle(select)
            for i in range(len(select)):
                f = minbias_files[select[i]]
                td = TrainData()
                td.readFromFile(f)
                arr=td.transferFeatureListToNumpy()[0]
                eventshere+=arr.shape[0]
                allpu.append(arr)
                if eventshere >= nevents:
                    break
                del td
            allpu = np.concatenate(allpu,axis=0)
            allpu = allpu[:nevents]
            np.random.shuffle(allpu)
            energy+=allpu
        
        pu_en = np.sum(energy) - befpu_en
        print('PU '+str(nPU)+ ' PU energy '+str(pu_en)+' signal en '+str(befpu_en))
        return energy
        
    
    
    def read_and_project(self, filename, weighterobjects, istraining, readPU):
        
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
        
        #create minbias and signal
        #rechit_energy = np.tile(rechit_energy,[2,1])
        issignal = np.array([0.,1.], dtype='float32')
        issignal = np.reshape(issignal, [2,1])
        issignal = np.tile(issignal, [nevents,1])
        #rechit_energy *= issignal
        
        if readPU:
            print('adding PU')
            rechit_energy = self.addPU(rechit_energy, rechit_energy.shape[0], 00)
            print('PU done')
        rechit_energy = np.reshape(rechit_energy, [-1, nofEELayers, etasegments, Ncalowedges]) #nofEELayers, etasegments, Ncalowedges
        print(rechit_energy.shape)
        rechit_energy = rechit_energy.transpose((0,2,3,1))
        print(rechit_energy.shape)
        #exit()
        
        #maybe use layer number as colour modifier here?
        
        colors = np.array(range(14),dtype='float')/14. #layer colours
        colors = cm.rainbow(colors)
        colors = colors[:,:-1]
        colors = np.reshape(colors, [1,1,1,14,3])
        rechit_energy = np.expand_dims(rechit_energy, axis=-1)
        
        print(rechit_energy.shape)
        print(colors.shape)
        
        norm = np.array([[[[
            [0.00010236],
            [0.0002501 ],
            [0.00041541],
            [0.00054669],
            [0.00065345],
            [0.00072501],
            [0.00074395],
            [0.00071629],
            [0.00067139],
            [0.00060171],
            [0.00052976],
            [0.00047216],
            [0.00039632],
            [0.00024775],
            ]]]],dtype='float32')
        
        rechit_energy = np.array(rechit_energy/norm*colors, dtype='float32')
        
        all = np.sum(rechit_energy, axis=3)
        
        ##rechit energy: B x eta x phi x layerEn (14)
        #
        #first_projection = np.sum(rechit_energy[:,:,:,0:],axis=-1,keepdims=True)/2.#/1.5816298e-05/0.65174264
        #sec_projection   = np.sum(rechit_energy[:,:,:,7:10],axis=-1,keepdims=True)#/3.0695286e-05/0.9940545
        #third_projection = np.sum(rechit_energy[:,:,:,10:],axis=-1,keepdims=True)#/1.6368242e-05/1.3476628
        #
        #all = first_projection #np.concatenate([first_projection,sec_projection,third_projection], axis=-1)
        #

        print(all.shape)
        
        
        return all, issignal
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        
        all, issignal = self.read_and_project(filename, weighterobjects, istraining,True)
        
        #eal with phi modulo. it is 120/2pi, so 0.4 in phi should be enough, so 8 extra repitions
        all = np.concatenate([all, all[:,:,:8,:]],axis=2)
        
        debug=True
        if debug:
            #event=1
            for event in range(50):
                evtsforplot = np.sum(all[event:event+1], axis=0)
                import matplotlib.pyplot as plt
                fig,ax =  plt.subplots(1,1)
                maxcol = np.max(evtsforplot)
                ax.imshow(evtsforplot/maxcol)
                #print('max energy '+str(maxcol))
                fig.savefig("event_displ"+str(event)+".pdf")
                plt.close()
        
        return [all],[issignal],[]
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(predicted[0].transpose(), 
                                             names='prob_isLLP')
        
        array2root(out, outfilename, 'tree')
        
