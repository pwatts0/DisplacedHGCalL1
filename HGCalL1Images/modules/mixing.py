
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
    #print(select)
    #open them
    #take nPU random events
    inarrs=[]
    for i in range(nfiles):
        file = minbias_files[select[i]]
        print('mixing '+str(select[i]))
        fileTimeOut(file,10)
        #check if file is valid
        try:
            f=ROOT.TFile.Open(file)
            f.Get("B4")
        except:
            nfiles+=1 #just take the next one
            continue
        
        tree = uproot.open(file)["B4"]
        arr = tonumpy(tree["rechit_energy"].array() )
        arr = np.expand_dims(arr, axis=0)# 1 x nev x rh
        inarrs.append(arr)
    
    allarr = np.concatenate(inarrs, axis=0) #nfiles x nev x rh
    allarr = np.reshape(allarr, [allarr.shape[0]*allarr.shape[1],allarr.shape[2]])
    
    #print(allarr.shape)
    evtarrs=[]
    for ev in range(nevents):
        idx = np.random.randint(allarr.shape[0], size=nPU)
        evt = allarr[idx]
        evt = np.sum(evt,axis=0, keepdims=True)
        #print(evt.shape)
        evtarrs.append(evt)
    
    return np.concatenate(evtarrs,axis=0)
        
        
def premixfile(allfiles,neventstotal,nPU,nfilespremix=5,eventsperround=100): 

    nevents=0 
    filearr=[]

    while nevents<neventstotal:         
        filearr.append(readPU(allfiles, nevents=eventsperround, nfiles=nfilespremix, nPU=nPU))
        nevents+=eventsperround
    filearr = np.concatenate(filearr,axis=0)
    if len(filearr) > neventstotal:
        filearr = filearr[0:neventstotal]
    return filearr
    