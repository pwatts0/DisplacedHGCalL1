
import numpy as np
import uproot
import glob

def tonumpy(inarr):
    return np.array(list(inarr), dtype='float32')


def getAndCheck(path, check=True):
    minbias_files= glob.glob(path+"/*.root")
    from datastructures import TrainData_calo
    td=TrainData_calo()
    outfiles=[]
    for f in minbias_files:
        print('check '+f)
        if (not check) or td.fileIsValid(f):
            outfiles.append(f)
        else:
            print('file '+f+' broken')
    
    return outfiles


def readPU_lookup():
    pass

#makes 
def readPU(minbias_files, nevents=50, nfiles=5, nPU=200):
    
    from DeepJetCore.TrainData import fileTimeOut
    import ROOT
    
    select = np.array(range(len(minbias_files)))
    np.random.shuffle(select)
    if len(select)<nfiles:
        nfiles=len(select)
        print("mixing.readPU: warning: less PU files available than requested - falling back")
        print(nfiles)
    #print(select)
    #open them
    #take nPU random events
    inarrs=[]
    i=0
    while len(inarrs) < nfiles:
        file = minbias_files[select[i]]
        i+=1
        #print('mixing: get data '+str(select[i])+' to go '+str(nfiles-i-1))
        fileTimeOut(file,10)
        #check if file is valid
        try:
            f=ROOT.TFile.Open(file)
            f.Get("B4")
        except:
            continue
        
        ramfile=file
        try:
            tree = uproot.open(ramfile)["B4"]
            arr = tonumpy(tree["rechit_energy"].array() )
            #print('arr',arr.shape)
            #arr = np.expand_dims(arr, axis=0)# 1 x nev x rh
            #print('arr2',arr.shape, ramfile)
            inarrs.append(arr)
        except:
            continue
    
    allarr = np.concatenate(inarrs, axis=0) # nfiles*nev x rh
    #allarr = np.reshape(allarr, [allarr.shape[0]*allarr.shape[1],allarr.shape[2]])
    
    #print(allarr.shape)
    #print('mixing events')
    evtarrs=[]
    for ev in range(nevents):
        idx = np.random.randint(allarr.shape[0], size=nPU)
        evt = allarr[idx]
        evt = np.sum(evt,axis=0, keepdims=True)
        #print(evt.shape)
        evtarrs.append(evt)
        
    evts = np.concatenate(evtarrs,axis=0)
    #print('mixed '+str(evts.shape))
    return evts
        
        
def premixfile(allfiles,neventstotal,nPU,nfilespremix=5,eventsperround=100,seed=0): 

    nevents=0 
    filearr=[]
    np.random.seed(seed)
    
    if eventsperround>neventstotal:
        eventsperround=neventstotal

    while nevents<neventstotal:         
        filearr.append(readPU(allfiles, nevents=eventsperround, nfiles=nfilespremix, nPU=nPU))
        nevents+=eventsperround
    filearr = np.concatenate(filearr,axis=0)
    if len(filearr) > neventstotal:
        filearr = filearr[0:neventstotal]
    return filearr
    