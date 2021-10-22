#from __future__ import division                                                                                                                    
import json
import numpy as np
import os, sys
import h5py
import time
import datetime
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer

from DATAutils import *
from NNutils import *
from PLOTutils import *

###########################################
parser = argparse.ArgumentParser()    
parser.add_argument('-j', '--jsonfile'  , type=str, help="json file", required=True)
args = parser.parse_args()

seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

with open(args.jsonfile, 'r') as jsonfile:
        config_json = json.load(jsonfile)

columns_training = config_json["features"]

N_Bkg   = config_json["N_Bkg"]
N_D     = N_Bkg

#### Architecture: #########################
BSM_architecture = config_json["BSMarchitecture"]
BSM_wclip        = config_json["BSMweight_clipping"]

patience     = config_json["patience"]
total_epochs = config_json["epochs"]
#mass_cut     = config_json["Mcut"]

##### define output path ######################
OUTPUT_PATH    = config_json["output_directory"]
OUTPUT_FILE_ID = '/Toy5D_seed'+str(seed)+'_patience'+str(patience)

# do not run the job if the toy label is already in the folder
if os.path.isfile("%s/%s_t.txt" %(OUTPUT_PATH, OUTPUT_FILE_ID)):
        exit()

# nuisance values ###########################                                                                                                               
correction       = config_json["correction"] # 'NORM', 'SHAPE', ''

# shape effects
shape_sigma      = np.array(config_json["shape_nuisances_sigma"])
shape_generation = np.array(config_json["shape_nuisances_data"]) # units of sigma
shape_generation = shape_generation*shape_sigma # absolute vlaues
shape_reference  = np.array(config_json["shape_nuisances_reference"]) # units of sigma 
shape_reference  = shape_reference*shape_sigma # absolute vlaues
shape_auxiliary  = []
for i in range(len(shape_sigma)):
    if shape_sigma[i]:
        shape_auxiliary.append(np.random.normal(shape_generation[i], shape_sigma[i], size=(1,))[0])
    else:
        shape_auxiliary.append(0)
shape_auxiliary = np.array(shape_auxiliary)
shape_dictionary_list = config_json["shape_dictionary_list"]

# global normalization
norm_sigma      = config_json["norm_nuisances_sigma"]
norm_generation = config_json["norm_nuisances_data"] # units of sigma                                                                                 
norm_generation = norm_generation*norm_sigma # absolute vlaues
norm_reference  = config_json["norm_nuisances_reference"] # units of sigma  
norm_reference  = norm_reference*norm_sigma # absolute vlaues  
if norm_sigma:
        norm_auxiliary  = np.random.normal(norm_generation, norm_sigma, size=(1,))[0]
else:
        norm_auxiliary = 0

# cross section normalization
csec_nuisances_data       = config_json["csec_nuisances_data"] # units of sigma  
csec_nuisances_reference  = config_json["csec_nuisances_reference"] # units of sigma  
csec_nuisances_sigma      = config_json["csec_nuisances_sigma"]


##### Read data ###############################
INPUT_PATH_REF = '/eos/user/g/ggrosso/PhD/NOTEBOOKS/HZZ4L/CMS_analysis/h5_files_v2/'
if 'Z1Z2DeltaPhi' in columns_training:
        INPUT_PATH_REF = '/eos/user/g/ggrosso/PhD/NOTEBOOKS/HZZ4L/CMS_analysis/h5_files_v3/'

feature_dict   = {
        'weight_REF': np.array([]),
        'weight_DATA': np.array([]),
        'l1Id': np.array([]),
        'l2Id': np.array([]),
        'l3Id': np.array([]),
        'l4Id': np.array([]),
}
for key in columns_training:
        feature_dict[key] = np.array([])
for process in bkg_list:
        f = h5py.File(INPUT_PATH_REF+process+'.h5', 'r')
        #cross section uncertainty factor to generate the reference (exponential parametrization, usually 1)
        cross_sx_nu_D = np.exp(csec_nuisances_data[process]*csec_nuisances_sigma[process]) 
        cross_sx_nu_R = np.exp(csec_nuisances_reference[process]*csec_nuisances_sigma[process])
        w = np.array(f.get('weight'))
        feature_dict['weight_REF']  = np.append(feature_dict['weight_REF'],  w*cross_sx_nu_R)
        feature_dict['weight_DATA'] = np.append(feature_dict['weight_DATA'], w*cross_sx_nu_D)
        feature_dict['l1Id']        = np.append(feature_dict['l1Id'], np.array(f.get('l1Id')))
        feature_dict['l2Id']        = np.append(feature_dict['l2Id'], np.array(f.get('l2Id')))
        feature_dict['l3Id']        = np.append(feature_dict['l3Id'], np.array(f.get('l3Id')))
        feature_dict['l4Id']        = np.append(feature_dict['l4Id'], np.array(f.get('l4Id')))
        for key in columns_training:
                feature_dict[key] = np.append(feature_dict[key], np.array(f.get(key)))
        f.close()
        print('process: %s --> number of simulations: %i, yield: %f'%(process, w.shape[0], np.sum(w)))

#select only 4mu final state
mask = (np.abs(feature_dict['l1Id'])==13)*(np.abs(feature_dict['l2Id'])==13)*(np.abs(feature_dict['l3Id'])==13)*(np.abs(feature_dict['l4Id'])==13)
for key in list(feature_dict.keys()):
    if key in ['l1Id', 'l2Id', 'l3Id', 'l4Id']: continue
    var = feature_dict[key]
    var = var[mask]
    feature_dict[key] = var
for key in ['l1Id', 'l2Id', 'l3Id', 'l4Id']:
    var = feature_dict[key]
    var = var[mask]
    feature_dict[key] = var

weight_sum_R = np.sum(feature_dict['weight_REF'])
weight_sum_D = np.sum(feature_dict['weight_DATA'])

REF     = np.stack([feature_dict[key] for key in list(columns_training)], axis=1)
W_REF   = feature_dict['weight_REF']

#unweighting DATA
N_REF  = REF.shape[0]
weight = feature_dict['weight_DATA']
f_MAX  = np.max(weight) 

indeces  = np.arange(weight.shape[0])
np.random.shuffle(indeces)
DATA     = np.array([])
DATA_idx = np.array([])
print('normalization effect on N_D: %f'%(np.exp(norm_generation*norm_sigma)))
print('cross section effect on N_D: %f'%(weight_sum_D/weight_sum_R))
N_DATA   = np.random.poisson(lam=N_Bkg*weight_sum_D/weight_sum_R*np.exp(norm_generation*norm_sigma), size=1)[0]
print('N_Bkg: '+str(N_Bkg))
print('N_Bkg_Pois: '+str(N_DATA))
if N_REF<N_DATA:
        print('Cannot produce %i events; only %i available'%(N_DATA, N_REF))
        exit()
counter = 0
while DATA.shape[0]<N_DATA:
        i = indeces[counter]
        x = REF[i:i+1, :]
        f = weight[i]
        if f<0:
                DATA_idx = np.append(DATA_idx, i) 
                counter+=1
                continue
        r = f/f_MAX
        if r>=1:
                if DATA.shape[0]==0:
                        DATA = x
                        DATA_idx = np.append(DATA_idx, i)
                else:
                        DATA = np.concatenate((DATA, x), axis=0)
                        DATA_idx = np.append(DATA_idx, i)
        else:
                u = np.random.uniform(size=1)
                if u<= r:
                        if DATA.shape[0]==0:
                                DATA = x
                                DATA_idx = np.append(DATA_idx, i)
                        else:
                                DATA = np.concatenate((DATA, x), axis=0)
                                DATA_idx = np.append(DATA_idx, i)
        counter+=1
        if counter>=REF.shape[0]:
                print('End of file')
                N_DATA = DATA.shape[0]
                break
weight  = np.delete(W_REF, DATA_idx, 0)
REF     = np.delete(REF, DATA_idx, 0)
# correct weights for the factor lost due to sampling out the toy
weight = weight *weight_sum_D/np.sum(weight)

'''
# display training variables to debug
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')
i=0
for key in columns_training:
    fig = plt.figure(figsize=(10, 12))
    fig.patch.set_facecolor('white')
    ax1 = fig.add_axes([0., 0.33, 0.99, 0.6])
    # REF
    var = REF[:, i]
    w   = weight
    bins =  bins_code[key]
    plt.hist(var, weights=w, color='#fed976', ec='#b10026', label='Reference', bins=bins)
    hR  = plt.hist(var, weights=w, color='#fed976', ec='#b10026', alpha=0., bins=bins)
    hR1 = plt.hist(var           , color='#fed976', ec='#b10026', alpha=0., bins=bins)
    # DATA
    hD = plt.hist(DATA[:, i], histtype='step', lw=3, color='black', bins=bins)
    x  = 0.5*(hD[1][1:]+hD[1][:-1])
    y  = hD[0]
    plt.errorbar(x, y, yerr=np.sqrt(y), color='black', lw=3, label='Data', ls='')
    plt.ylim(0.01, 1.2*np.max(hD[0]))
    plt.yscale('log')
    plt.legend(fontsize=16)
    plt.ylabel('Counts', fontsize=20)
    plt.yticks(fontsize=16)
    plt.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
    ax2 = fig.add_axes([0., 0., 0.99, 0.3])
    ax2.errorbar(x, hD[0]/hR[0], yerr=hD[0]/hR[0]*np.sqrt(1./hD[0] + 1./hR1[0]), color='black', marker='o', ls='', lw=3)
    ax2.plot(x, np.ones_like(x), ls='--', color='grey', lw=2)
    plt.grid()
    plt.ylabel('Data/Reference', fontsize=20)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel(xlabel_code[key], fontsize=20)
    plt.ylim(0, 2)
    plt.show()
    plt.close()
    i+=1
exit()
'''
############################################
feature = np.concatenate((REF, DATA), axis=0)
weights = np.concatenate((weight, np.ones(DATA.shape[0])), axis=0)
target  = np.concatenate((np.zeros(REF.shape[0]), np.ones(DATA.shape[0])), axis=0)
target  = np.stack((target, weights), axis=1) 

#standardize dataset #######################
for j in range(feature.shape[1]):
    vec  = feature[:, j]
    mean = mean_bkg[columns_training[j]]#np.mean(vec)                                                                    
    std  = std_bkg[columns_training[j]]#np.std(vec) 
    if np.min(vec) < 0:
        vec = vec- mean
        vec = vec *1./ std
    elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.                                                        
        vec = vec *1./ mean
    feature[:, j] = vec

#### training tau ###############################
print('training tau')
batch_size = feature.shape[0]
input_shape = (None, BSM_architecture[0])
BSMfinder  = NPLM_imperfect(input_shape, 
                            NU_S=shape_reference, NUR_S=shape_reference, NU0_S=shape_auxiliary, SIGMA_S=shape_sigma, 
                            NU_N=norm_reference, NUR_N=norm_reference, NU0_N=norm_auxiliary, SIGMA_N=norm_sigma, 
                            shape_dictionary_list=shape_dictionary_list,
                            BSMarchitecture=BSM_architecture, BSMweight_clipping=BSM_wclip, correction=correction,
                            train_nu=True, train_f=True)

BSMfinder.compile(loss = NPLM_Imperfect_Loss,  optimizer = 'adam')
hist = BSMfinder.fit(feature, target, batch_size=batch_size, epochs=total_epochs, verbose=False)
print('End training ')
##### OUTPUT ###############################
# test statistic                                                                                                     
loss = np.array(hist.history['loss'])
final_loss = loss[-1]
t_OBS      = -2*final_loss
print('tau: %f'%(t_OBS))
# save t                                                                                                               
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_tau.txt'
out   = open(log_t,'w')
out.write("%f\n" %(t_OBS))
out.close()

# write the loss history                                       
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_TAU_history'+str(patience)+'.h5'
f           = h5py.File(log_history,"w")
epoch       = np.array(range(total_epochs))
keepEpoch   = epoch % patience == 0
f.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
for key in list(hist.history.keys()):
    monitored = np.array(hist.history[key])
    print('%s: %f'%(key, monitored[-1]))
    f.create_dataset(key, data=monitored[keepEpoch],   compression='gzip')
f.close()

# save the model                                                                                                                                
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+'_TAU_weights.h5'
BSMfinder.save_weights(log_weights)

if correction == '':
        print('Correction not required. Exit.')
        exit()
#### training Delta #############################
print('Training Delta')                                                                
total_epochs_d = 2*total_epochs
batch_size = feature.shape[0]
input_shape = (None, BSM_architecture[0])
BSMfinder  = NPLM_imperfect(input_shape,
                            NU_S=shape_reference, NUR_S=shape_reference, NU0_S=shape_auxiliary, SIGMA_S=shape_sigma,
                            NU_N=norm_reference, NUR_N=norm_reference, NU0_N=norm_auxiliary, SIGMA_N=norm_sigma,
                            shape_dictionary_list=shape_dictionary_list,
                            BSMarchitecture=BSM_architecture, BSMweight_clipping=BSM_wclip, correction=correction,
                            train_nu=True, train_f=False)

opt  = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00000001)
BSMfinder.compile(loss = NPLM_Imperfect_Loss,  optimizer = opt)
hist = BSMfinder.fit(feature, target, batch_size=batch_size, epochs=total_epochs_d, verbose=False)
print('End training ')
##### OUTPUT ################################  
# test statistic                                                                                                                                            
loss = np.array(hist.history['loss'])
final_loss = loss[-1]
t_OBS      = -2*final_loss
print('Delta: %f'%(t_OBS))

# save t                                                                                                                                                   
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_Delta.txt'
out   = open(log_t,'w')
out.write("%f\n" %(t_OBS))
out.close()

# write the loss history                                                                                                                           
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_DELTA_history'+str(patience)+'.h5'
f           = h5py.File(log_history,"w")
epoch       = np.array(range(total_epochs_d))
keepEpoch   = epoch % patience == 0
f.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
for key in list(hist.history.keys()):
    monitored =np.array(hist.history[key])
    print('%s: %f'%(key, monitored[-1]))
    f.create_dataset(key, data=monitored[keepEpoch],   compression='gzip')
f.close()

# save the model 
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+'_DELTA_weights.h5'
BSMfinder.save_weights(log_weights)
