import numpy as np

csec_nuisances_sigma = {
    'AllData_ZX_redTree_2018': 0.3,
    'ggTo2e2tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo4mu_Contin_MCFM701_redTree_2018': 0,
    'ggTo2mu2tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo4tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo2e2mu_Contin_MCFM701_redTree_2018': 0,
    'ggTo4e_Contin_MCFM701_redTree_2018': 0,
    'ZZTo4lext_redTree_2018_0': 0,
    'ZZTo4lext_redTree_2018_1': 0,
    'ZZTo4lext_redTree_2018_2': 0,
    'ZZTo4lext_redTree_2018_3': 0,
}

csec_nuisances_reference = {
    'AllData_ZX_redTree_2018': 0,
    'ggTo2e2tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo4mu_Contin_MCFM701_redTree_2018': 0,
    'ggTo2mu2tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo4tau_Contin_MCFM701_redTree_2018': 0,
    'ggTo2e2mu_Contin_MCFM701_redTree_2018': 0,
    'ggTo4e_Contin_MCFM701_redTree_2018': 0,
    'ZZTo4lext_redTree_2018_0': 0,
    'ZZTo4lext_redTree_2018_1': 0,
    'ZZTo4lext_redTree_2018_2': 0,
    'ZZTo4lext_redTree_2018_3': 0,
}

#### Z+X uncertainty
weights_file = '/eos/user/g/ggrosso/PhD/NOTEBOOKS/HZZ4L/CMS_analysis/PARAMETRIC_Z+X/QUADRATIC_1Dpatience100_epochs10000_layers1_50_50_50_1_actrelu/model_weights900_fullbatch.h5'
sigma = 0.3#weights_file.split('sigma', 1)[1]
#sigma = float(sigma.split('_', 1)[0])
scale_list=np.array([-3, -1, 1, 3])*sigma#weights_file.split('sigma', 1)[1]
#scale_list=scale_list.split('_patience', 1)[0]
#scale_list=np.array([float(s) for s in scale_list.split('_')[1:]])*sigma
shape_std = np.std(scale_list)
activation= weights_file.split('act', 1)[1]
wclip=None
if 'wclip' in weights_file:
    activation=activation.split('_', 1)[0]
    wclip= weights_file.split('wclip', 1)[1]
    wclip = float(wclip.split('/', 1)[0])
else:
    activation=activation.split('/', 1)[0]
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
    print('Unrecognized number of degree for polynomial parametric net in file: \n%s'%(weights_file))
    exit()
ZX_unc_parNN = { 'poly_degree'   : poly_degree,
                 'architectures' : [architecture for i in range(poly_degree)],
                 'wclips'       : [wclip for i in range(poly_degree)],
                 'activation'    : activation,
                 'shape_std'     : shape_std,
                 'weights_file'  : weights_file
}

parNN_list = { 'AllData_ZX_redTree_2018': ZX_unc_parNN,
}
