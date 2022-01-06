import glob, os, time, math, h5py
import numpy as np
from scipy.stats import chi2, norm, uniform

def collect_txt(DIR_IN, suffix='t', files_prefix=[],  verbose=False):
    '''
    For each toy the function reads the .txt file where the final value for the variable t=-2*loss is saved. 
    It then associates a label to each toy.
    The array of the t values (tvalues) and the array of labels (files_id) are saved in an .h5 file.
    
    DIR_IN: directory where all the toys' outputs are saved
    DIR_OUT: directory where to save the .h5 output file
    
    The function returns the array of labels.
    '''
    dt = h5py.special_dtype(vlen=str)
    tvalues, files_id, seeds = np.array([]), np.array([]), np.array([])
    if len(files_prefix)>0:
        if verbose:
            print('use given files_prefix list')
    for fileIN in glob.glob("%s/*_%s.txt" %(DIR_IN, suffix) ):
        keep_file = True
        if len(files_prefix)>0:
            keep_file = False
            for f in files_prefix:
                if f in fileIN:
                    keep_file=True
                    break
            if not keep_file:
                continue
        f = open(fileIN)
        lines = f.readlines()
        file_id=  fileIN.split('/')[-1]
        seed = file_id.split('seed')[1]
        seed = int(seed.split('_')[0])

        file_id = file_id.replace('_%s.txt'%(suffix), '')
        f.close()
        if len(lines)==0:
            if verbose: print('empty')
            continue
        t = float(lines[0])
        if(np.isnan(np.array([t]))):
            if verbose: print('nan')
            t = -1
        print(t)
        tvalues  = np.append(tvalues, t)
        files_id = np.append(files_id, file_id)
        seeds    = np.append(seeds, seed)
    
    files_id=np.array(files_id, dtype=dt)
    if files_id.shape[0]==0:
        print('Empty folder')
    return tvalues, files_id, seeds

def save_txt_to_h5(tvalues, files_id, seeds, suffix, DIR_OUT, FILE_NAME=''):
    '''
    The function save the 1D-array of the test statistics (tvalues) in a .h5 file, 
    together with the id of each job (files_id) and the seeds.
    
    DIR_OUT: directory where to save the output file
    FILE_NAME: output file name
    suffix: label to be appended to the file_name
    '''
    log_file = '%s/%s_%s.h5'%(DIR_OUT,FILE_NAME, suffix)
    f = h5py.File(log_file, 'w')
    f.create_dataset('tvalues',  data=tvalues,  compression='gzip')
    f.create_dataset('files_id', data=files_id, compression='gzip')
    f.create_dataset('seeds',    data=seeds,    compression='gzip')
    f.close()
    print('Saved to file: %s'%(log_file))
    return

def collect_history(files_id, DIR_IN, suffix='t', key='loss', verbose=False):
    '''
    For each toy whose file ID is in the array files_id, 
    the function collects the history of the loss and saves t=-2*loss at the check points.
    
    files_id: array of toy labels 
    DIR_IN: directory where all the toys' outputs are saved
    
    The function returns a 2D-array with final shape (nr toys, nr check points).
    '''
    tdistributions_check =np.array([])
    cnt=0
    for file_id in files_id:
        history_file = DIR_IN+file_id+suffix+'_history.h5'
        if not os.path.exists(history_file):
            continue
        f = h5py.File(history_file, 'r')
        if not key in list(f.keys()):
            print('unknown key.')
            print('Available keys are:')
            print(f.keys())
            f.close()
            return []
        loss = np.array(f.get(key) )       
        if key == 'loss':
            loss = -2*loss
        loss = np.expand_dims(loss, axis=1)
        if not cnt:
            tdistributions_check = loss
        else:
            if loss.shape[0] != tdistributions_check.shape[0]: continue
            tdistributions_check = np.concatenate((tdistributions_check, loss), axis=1)
        cnt = cnt+1
        f.close()
    if verbose:
        print('Final history array shape')
        print('(nr toys, nr check points)')
        print(tdistributions_check.T.shape)
    return tdistributions_check.T

def save_history_to_h5(suffix, patience, tvalues_check, DIR_OUT, FILE_NAME='', seeds=[]):
    '''
    The function save the 2D-array of the loss histories in an .h5 file. 
    The keys are the epoch checkpoints the data are the set of values collected at that checkpoint.
    
    DIR_OUT: directory where to save the output file
    FILE_NAME: output file name
    suffix: label to be appended to the file_name
    patience: history rate
    '''
    nr_check_points = tvalues_check.shape[1]
    epochs_check = np.arange(1, nr_check_points+1)*patience
        
    log_file = '%s/%s_%s.h5'%(DIR_OUT, FILE_NAME, suffix)
    f = h5py.File(log_file,"w")
    if len(seeds)!=0:
        f.create_dataset('seeds', data=np.array(seeds), compression='gzip')
    for i in range(nr_check_points):
        f.create_dataset('%s'%(epochs_check[i]), data=tvalues_check[:, i], compression='gzip')
    f.close()
    print('Saved to file: %s'%(log_file))
    return

def collect_weights(DIR_IN, suffix='_weights'):
    parameters = {}
    init = False
    for fileIN in glob.glob("%s/*_%s.h5" %(DIR_IN, suffix)):
        f = h5py.File(fileIN, 'r')
        for j in f:
            for k in f.get(j):
                for m in f.get(j).get(k):
                    if not init:
                        parameters[k+'_'+m[0]]= np.expand_dims(np.array(f.get(j).get(k).get(m)), axis=0)
                    else:
                        parameters[k+'_'+m[0]]= np.concatenate((parameters[k+'_'+m[0]], np.expand_dims(np.array(f.get(j).get(k).get(m)), axis=0)), axis=0)
        f.close()
        init=True
        
    return parameters

def Read_final_from_h5(DIR_IN, FILE_NAME, suffix):
    log_file = DIR_IN+FILE_NAME+suffix+'.h5'
    tvalues_check = np.array([])
    f = h5py.File(log_file,"r")
    t = f.get('tvalues')
    if 'seeds' in list(f.keys()):
        s = np.array(f.get('seeds'))
    elif 'files_id' in list(f.keys()):
        s = np.array([])
        l = np.array(f.get('files_id'))
        for label in l:
            seed =label.split('seed')[1]
            seed = int(seed.split('_')[0])
            s = np.append(s, seed)
    t = np.array(t)
    print(t.shape)
    f.close()
    return t, s

def Read_history_from_h5(DIR_IN, FILE_NAME, suffix):
    '''
    The function creates a 2D-array from a .h5 file.
    
    DIR_OUT: directory where to save the input file
    FILE_NAME: input file name
    suffix: label to be appended to the FILE_NAME
    
    The function returns a 2D-array with final shape (nr toys, nr check points). 
    '''
    log_file = DIR_IN+FILE_NAME+suffix+'.h5'
    f = h5py.File(log_file,"r")
    epochs_check  = [int(key) for key in list(f.keys())]
    tvalues_check = np.array([])
    for i in range(len(epochs_check)):
        t = f.get(str(epochs_check[i]))
        t = np.array(t)
        t = np.expand_dims(t, axis=1)
        if not i:
            tvalues_check = t
        else:
            tvalues_check = np.concatenate((tvalues_check, t), axis=1)
    f.close()
    print('Output shape:')
    print(tvalues_check.shape)
    return tvalues_check

### tools to compute the Kolmogorov-Smirnov test statistic
def KSDistanceToUniform(sample):
    Ntrials      = sample.shape[0]
    ECDF         = np.array([i*1./Ntrials for i in np.arange(Ntrials+1)])
    sortedsample = np.sort(sample)
    sortedsample = np.append(0, np.sort(sample))
    KSdist       = 0
    if Ntrials==1:
        KSdist = np.maximum(1-sortedsample[1], sortedsample[1])
    else:
        KSdist = np.max([np.maximum(np.abs(sortedsample[i+1]-ECDF[i]), np.abs(sortedsample[i+1]-ECDF[i+1])) for i in np.arange(Ntrials)])
    return KSdist

def KSTestStat(data, ndof):
    sample = chi2.cdf(data, ndof)
    KSdist = KSDistanceToUniform(sample)
    return KSdist

def GenUniformToy(Ntrials):
    sample = np.random.uniform(size=(Ntrials,))
    KSdist = KSDistanceToUniform(sample)
    return KSdist

def GetTSDistribution(Ntrials, Ntoys=1000):
    KSdistDistribution = []
    for i in range(Ntoys):
        KSdist = GenUniformToy(Ntrials)
        KSdistDistribution.append(KSdist)
    return np.array(KSdistDistribution)

def pvalue(KSTestStat_Value, KSdistDistribution):
    pval_right=np.sum(1*(KSdistDistribution>KSTestStat_Value))*1./KSdistDistribution.shape[0]
    return pval_right

def GenToyFromEmpiricalPDF(sample):
    Ntrials = sample.shape[0]
    indeces = np.random.randint(low=0, high=Ntrials, size=(Ntrials,))
    toy     = np.array([sample[indeces[i]] for i in range(Ntrials)])
    return toy

def KS_test(sample, dof, Ntoys=100000):
    Ntrials            = sample.shape[0]
    KSTestStat_Value   = KSTestStat(sample, dof)
    KSdistDistribution = GetTSDistribution(Ntrials=Ntrials, Ntoys=Ntoys)
    pval               = pvalue(KSTestStat_Value, KSdistDistribution)
    return pval

def ErrorOnKSTestStat(sample, dof, Ntrials, Ntoys=100000):
    KS_out = []
    for _ in range(Ntrials):
        
        tmp_idx = np.random.choice(sample.shape[0], sample.shape[0])
        tmp_idx = tmp_idx.astype(int)
        tmp_sample = sample[tmp_idx]
        KS_out.append(KS_test(tmp_sample, dof=dof, Ntoys=Ntoys))
    KS_out = np.array(KS_out)
    plt.hist(KS_out)
    plt.show()
    error  = np.sqrt(np.var(KS_out))
    print('error: %f'%(error))
    return error, KS_out
###################################################

def Extract_Tail(tvalues_check, patience, cut=95, verbose=False):
    tail_distribution = np.array([])
    normal_distribution = np.array([])
    epochs_check = []
    size = tvalues_check.shape[0]
    nr_check_points = tvalues_check.shape[1]
    for i in range(nr_check_points):
        epoch_check = patience*(i+1)
        epochs_check.append(epoch_check)
        
    for i in range(tvalues_check.shape[1]):
        tvalues = np.sort(tvalues_check[:, i])
        percentile_cut = int(cut*0.01*size)
        bulk_distribution_i = tvalues[:percentile_cut]
        bulk_distribution_i = np.expand_dims(bulk_distribution_i, axis=1)
        tail_distribution_i = tvalues[percentile_cut:]
        tail_distribution_i = np.expand_dims(tail_distribution_i, axis=1)
        if not i:
            tail_distribution = tail_distribution_i
            bulk_distribution = bulk_distribution_i
        else:
            tail_distribution = np.concatenate((tail_distribution, tail_distribution_i), axis=1)
            bulk_distribution = np.concatenate((bulk_distribution, bulk_distribution_i), axis=1)
    if verbose:
        print('Tail distributions shape')
        print(tail_distribution.shape)
        print('Bulk distributions shape')
        print(bulk_distribution.shape)
    return tail_distribution, bulk_distribution