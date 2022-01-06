import os
import json
import argparse
import numpy as np
import glob
import os.path
import time
from config_utils import parNN_list

OUTPUT_DIRECTORY = './'

def create_config_file(config_table, OUTPUT_DIRECTORY):
    with open('%s/config.json'%(OUTPUT_DIRECTORY), 'w') as outfile:
        json.dump(config_table, outfile, indent=4)
    return '%s/config.json'%(OUTPUT_DIRECTORY)

# configuration dictionary
config_json = {
    "N_Ref"   : 200000,
    "N_Bkg"   : 2000,
    "N_Sig"   : 0,#10,
    "output_directory": OUTPUT_DIRECTORY,
    "shape_nuisances_id":        ['scale'],
    "shape_nuisances_data":      [1],#[0],
    "shape_nuisances_reference": [0],
    "shape_nuisances_sigma":     [0.05], 
    "shape_dictionary_list":     [parNN_list['scale']],
    "norm_nuisances_data":       0,
    "norm_nuisances_reference":  0,
    "norm_nuisances_sigma":      0.05,
    "epochs_tau": 3000,
    "patience_tau": 100,
    "epochs_delta": 3000,
    "patience_delta": 100,
    "BSMarchitecture": [1,4,1],
    "BSMweight_clipping": 9, 
    "correction": "SHAPE", # "SHAPE", "NORM", ""
}

# list process normalization generation values from shape uncertainties generation values
for i in range(len(config_json["shape_nuisances_id"])):
    key = config_json["shape_nuisances_id"][i]

# check for errors in the config_json dictionary
n_dimensions = config_json["BSMarchitecture"][0]
if config_json["BSMarchitecture"][0] != n_dimensions:
    print('Error: number of training dimensions and input layer size do not match.')
    exit()
if len(config_json["shape_nuisances_sigma"])!=len(config_json["shape_dictionary_list"]):
    print('Error: length of "shape_nuisances_sigma" and "shape_dictionary_list" must be the same.')
    exit()
if len(config_json["shape_nuisances_sigma"])!=len(config_json["shape_nuisances_data"]):
    print('Error: length of "shape_nuisances_sigma" and "shape_nuisances_data" must be the same.')
    exit()
if len(config_json["shape_nuisances_sigma"])!=len(config_json["shape_nuisances_reference"]):
    print('Error: length of "shape_nuisances_sigma" and "shape_nuisances_reference" must be the same.')
    exit()
if config_json["correction"]=='SHAPE' and not len(config_json["shape_dictionary_list"]):
    print('Error: correction is SHAPE but not specified "shape_dictionary_list" in the configuration dictionary.')
    exit()

# add details about the experiment set up to the folder name
correction_details = config_json["correction"]
if config_json["correction"]=='SHAPE':
    correction_details += str(len(config_json["shape_dictionary_list"]))+'_'
    for i in range(len(config_json["shape_nuisances_id"])):
        key = config_json["shape_nuisances_id"][i]
        if config_json["shape_nuisances_data"][i] !=0:
            correction_details += 'nu'+key+str(config_json["shape_nuisances_data"][i])+'_'
if config_json["correction"]=='NORM' or config_json["correction"]=='SHAPE':
    if config_json["correction"]=='NORM':
        correction_details += '_'
    correction_details += 'nuN'+ str(config_json["norm_nuisances_data"])+'_'
ID =str(n_dimensions)+'D/'+correction_details+'Nbkg'+str(config_json["N_Bkg"])+'_Nsig'+str(config_json["N_Sig"]) 
ID+='_epochsTau'+str(config_json["epochs_tau"])+'_epochsDelta'+str(config_json["epochs_delta"])
ID+='_arc'+str(config_json["BSMarchitecture"]).replace(', ', '_').replace('[', '').replace(']', '')+'_wclip'+str(config_json["BSMweight_clipping"])


#### launch python script ###########################
if __name__ == '__main__':
    parser   = argparse.ArgumentParser()
    parser.add_argument('-p','--pyscript', type=str, help="name of python script to execute", required=True)
    parser.add_argument('-l','--local',    type=int, help='if to be run locally',             required=False, default=0)
    parser.add_argument('-t', '--toys',    type=int, help="number of toys to be processed",   required=False, default = 100)
    args     = parser.parse_args()
    ntoys    = args.toys
    pyscript = args.pyscript
    config_json['pyscript'] = pyscript
    
    pyscript_str = pyscript.replace('.py', '')
    pyscript_str = pyscript_str.replace('_', '/')
    config_json["output_directory"] = OUTPUT_DIRECTORY+'/'+pyscript_str+'/'+ID
    if not os.path.exists(config_json["output_directory"]):
        os.makedirs(config_json["output_directory"])
    
    json_path = create_config_file(config_json, config_json["output_directory"])
    if args.local:
        print('!!! Be sure you sourced /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh !!!')
        print('!!! or activate your personal environment before                                             !!!')
        os.system("python %s/%s -j %s" %(os.getcwd(), pyscript, json_path))
    else:
        label = "folder-log-jobs"
        os.system("mkdir %s" %label)
        for i in range(ntoys):        
            # src file
            script_src = open("%s/%i.src" %(label, i) , 'w')
            script_src.write("#!/bin/bash\n")
            script_src.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh\n")
            script_src.write("python %s/%s -j %s" %(os.getcwd(), args.pyscript, json_path))
            script_src.close()
            os.system("chmod a+x %s/%i.src" %(label, i))
            # condor file
            script_condor = open("%s/%i.condor" %(label, i) , 'w')
            script_condor.write("executable = %s/%i.src\n" %(label, i))
            script_condor.write("universe = vanilla\n")
            script_condor.write("output = %s/%i.out\n" %(label, i))
            script_condor.write("error =  %s/%i.err\n" %(label, i))
            script_condor.write("log = %s/%i.log\n" %(label, i))
            script_condor.write("+MaxRuntime = 500000\n")
            script_condor.write('requirements = (OpSysAndVer =?= "CentOS7")\n')
            script_condor.write("queue\n")
            script_condor.close()
            # condor file submission
            os.system("condor_submit %s/%i.condor" %(label,i))
