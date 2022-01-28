import sys, os, time, datetime, h5py, json, argparse
import numpy as np
from scipy.stats import norm, expon, chi2, uniform, chisquare

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable

from NPLM.NNutils import *
from NPLM.PLOTutils import *

parser = argparse.ArgumentParser()    
parser.add_argument('-j', '--jsonfile'  , type=str, help="json file", required=True)
args = parser.parse_args()

#### set up parameters ###############################
with open(args.jsonfile, 'r') as jsonfile:
    config_json = json.load(jsonfile)
    
seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

#### statistics                                                                                                            
N_ref      = config_json["N_Ref"]
N_Bkg      = config_json["N_Bkg"]
N_Sig      = config_json["N_Sig"]
N_R        = N_ref
N_D        = N_Bkg

#### nuisance parameters configuration   
correction= config_json["correction"]
NU_S, NUR_S, NU0_S, SIGMA_S = [], [], [], []
NU_N, NUR_N, NU0_N, SIGMA_N = 0, 0, 0, 0
shape_dictionary_list = []

if not correction=='':
    SIGMA_N   = config_json["norm_nuisances_sigma"]
    NU_N      = config_json["norm_nuisances_data"]*SIGMA_N
    NUR_N     = config_json["norm_nuisances_reference"]*SIGMA_N
    NU0_N     = np.random.normal(loc=NU_N, scale=SIGMA_N, size=1)[0]

if correction=='SHAPE':
    SIGMA_S   = config_json["shape_nuisances_sigma"]
    NU_S      = [config_json["shape_nuisances_data"][i]*SIGMA_S[i] for i in range(len(SIGMA_S))]
    NUR_S     = [config_json["shape_nuisances_reference"][i]*SIGMA_S[i] for i in range(len(SIGMA_S))]
    NU0_S     = np.array([np.random.normal(loc=NU_S[i], scale=SIGMA_S[i], size=1)[0] for i in range(len(SIGMA_S))])
    shape_dictionary_list=config_json["shape_dictionary_list"]

#### training time                
total_epochs_tau   = config_json["epochs_tau"]
patience_tau       = config_json["patience_tau"]
total_epochs_delta = config_json["epochs_delta"]
patience_delta     = config_json["patience_delta"]

#### architecture                
BSMweight_clipping = config_json["BSMweight_clipping"]
BSMarchitecture    = config_json["BSMarchitecture"]
inputsize          = BSMarchitecture[0]
BSMdf              = compute_df(input_size=BSMarchitecture[0], hidden_layers=BSMarchitecture[1:-1])

##### define output path ######################
OUTPUT_PATH    = config_json["output_directory"]
OUTPUT_FILE_ID = '/seed'+str(seed)

#### build training samples ###################
Norm = NU_N
Scale= NU_S[0]
# data
N_Bkg_Pois  = np.random.poisson(lam=N_Bkg*np.exp(Norm), size=1)[0]
if N_Sig:
    N_Sig_Pois = np.random.poisson(lam=N_Sig*np.exp(Norm), size=1)[0]

featureData = np.random.exponential(scale=np.exp(1*Scale), size=(N_Bkg_Pois, 1))
if N_Sig:
    featureSig  = np.random.normal(loc=6.4, scale=0.16, size=(N_Sig_Pois,1))*np.exp(Scale)
    featureData = np.concatenate((featureData, featureSig), axis=0)
featureRef  = np.random.exponential(scale=1., size=(N_ref, 1))
feature     = np.concatenate((featureData, featureRef), axis=0)

# target                    
targetData  = np.ones_like(featureData)
targetRef   = np.zeros_like(featureRef)
weightsData = np.ones_like(featureData)
weightsRef  = np.ones_like(featureRef)*N_D*1./N_R
target      = np.concatenate((targetData, targetRef), axis=0)
weights     = np.concatenate((weightsData, weightsRef), axis=0)
target      = np.concatenate((target, weights), axis=1)

batch_size  = feature.shape[0]
inputsize   = feature.shape[1]

#### training TAU ###############################
tau = imperfect_model(input_shape=(None, inputsize),
                      NU_S=NU_S, NUR_S=NUR_S, NU0_S=NU0_S, SIGMA_S=SIGMA_S, 
                      NU_N=NU_N, NUR_N=NUR_N, NU0_N=NU0_N, SIGMA_N=SIGMA_N,
                      correction=correction, shape_dictionary_list=shape_dictionary_list,
                      BSMarchitecture=BSMarchitecture, BSMweight_clipping=BSMweight_clipping, train_f=True, train_nu=True)
print(tau.summary())
tau.compile(loss=imperfect_loss,  optimizer='adam')

t0=time.time()
hist_tau = tau.fit(feature, target, batch_size=batch_size, epochs=total_epochs_tau, verbose=False)
t1=time.time()
print('Training time (seconds):')
print(t1-t0)

# metrics                      
loss_tau  = np.array(hist_tau.history['loss'])

# test statistic                                         
final_loss = loss_tau[-1]
tau_OBS    = -2*final_loss
print('tau_OBS: %f'%(tau_OBS))

# save t                                                                                                               
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_TAU.txt'
out   = open(log_t,'w')
out.write("%f\n" %(tau_OBS))
out.close()

# save the training history                                       
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_TAU_history.h5'
f           = h5py.File(log_history,"w")
epoch       = np.array(range(total_epochs_tau))
keepEpoch   = epoch % patience_tau == 0
f.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
for key in list(hist_tau.history.keys()):
    monitored = np.array(hist_tau.history[key])
    print('%s: %f'%(key, monitored[-1]))
    f.create_dataset(key, data=monitored[keepEpoch],   compression='gzip')
f.close()

# save the model    
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+'_TAU_weights.h5'
tau.save_weights(log_weights)

#### training delta ###########################
delta = imperfect_model(input_shape=(None, inputsize),
                      NU_S=NU_S, NUR_S=NUR_S, NU0_S=NU0_S, SIGMA_S=SIGMA_S, 
                      NU_N=NU_N, NUR_N=NUR_N, NU0_N=NU0_N, SIGMA_N=SIGMA_N,
                      correction=correction, shape_dictionary_list=shape_dictionary_list,
                      BSMarchitecture=BSMarchitecture, BSMweight_clipping=BSMweight_clipping, train_f=False, train_nu=True)

print(delta.summary())
opt  = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0000001)
delta.compile(loss=imperfect_loss,  optimizer=opt)

t0=time.time()
hist_delta = delta.fit(feature, target, batch_size=batch_size, epochs=total_epochs_delta, verbose=False)
t1=time.time()
print('Training time (seconds):')
print(t1-t0)

# metrics                      
loss_delta  = np.array(hist_delta.history['loss'])

# test statistic                                            
final_loss   = loss_delta[-1]
delta_OBS    = -2*final_loss
print('delta_OBS: %f'%(delta_OBS))

# save t                  
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_DELTA.txt'
out   = open(log_t,'w')
out.write("%f\n" %(delta_OBS))
out.close()

# save the training history  
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_DELTA_history.h5'
f           = h5py.File(log_history,"w")
epoch       = np.array(range(total_epochs_delta))
keepEpoch   = epoch % patience_delta == 0
f.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
for key in list(hist_delta.history.keys()):
    monitored =np.array(hist_delta.history[key])
    print('%s: %f'%(key, monitored[-1]))
    f.create_dataset(key, data=monitored[keepEpoch],   compression='gzip')
f.close()

# save the model 
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+'_DELTA_weights.h5'
delta.save_weights(log_weights)
