import numpy as np
weights_file  = './LINEAR_Parametric_EXPO1D_batches_ref40000_bkg40000_sigmaS0.1_-1.0_-0.5_0.5_1.0_patience300_epochs30000_layers1_4_1_actrelu_model_weights9300.h5'

sigma = weights_file.split('sigmaS', 1)[1]                                                                                                                 
sigma = float(sigma.split('_', 1)[0])                                                                                                                        
scale_list=weights_file.split('sigma', 1)[1]                                                                                   
scale_list=scale_list.split('_patience', 1)[0]                                                                                                               
scale_list=np.array([float(s) for s in scale_list.split('_')[1:]])*sigma  

shape_std = np.std(scale_list)
activation= weights_file.split('act', 1)[1]
activation=activation.split('_', 1)[0]

wclip=None
if 'wclip' in weights_file:
    wclip= weights_file.split('wclip', 1)[1]
    wclip = float(wclip.split('/', 1)[0])

layers=weights_file.split('layers', 1)[1]
layers= layers.split('_act', 1)[0]
architecture = [int(l) for l in layers.split('_')]

poly_degree = 0
if 'LINEAR' in weights_file:
    poly_degree = 1
elif 'QUADRATIC' in weights_file:
    poly_degree = 2
elif 'CUBIC' in weights_file:
    poly_degree = 3
else:
    poly_degree = None
    
scale_parNN = { 
    'poly_degree'   : poly_degree,
    'architectures' : [architecture for i in range(poly_degree)],
    'wclips'        : [wclip for i in range(poly_degree)],
    'activation'    : activation,
    'shape_std'     : shape_std,
    'weights_file'  : weights_file
    }

parNN_list = { 
    'scale': scale_parNN
    }
