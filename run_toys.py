import os
import json
import argparse
import numpy as np
import glob
import os.path
import time
from DATAutils import *
from NUutils import *

OUTPUT_DIRECTORY = '/eos/user/g/ggrosso/PhD/HZZ4L/NPLM_imperfect/'

config_json = {
    "features": ['ZZMass'],
    "N_Bkg"   : 788,
    "N_Sig"   : 0,#40,
    "output_directory": OUTPUT_DIRECTORY,
    "shape_nuisances_id":        ['AllData_ZX_redTree_2018'],
    "shape_nuisances_data":      [1],#[0],
    "shape_nuisances_reference": [0],
    "shape_nuisances_sigma":     [0.3], 
    "shape_dictionary_list":     [parNN_list['AllData_ZX_redTree_2018'] ],
    "norm_nuisances_data":       0,
    "norm_nuisances_reference":  0,
    "norm_nuisances_sigma":      0.026+0.025, #luminosity+muID_and_RECO_eff
    "csec_nuisances_data":       csec_nuisances_reference,#null initialization
    "csec_nuisances_reference":  csec_nuisances_reference,
    "csec_nuisances_sigma":      csec_nuisances_sigma, 
    "epochs": 300000,
    "patience": 10000,
    "BSMarchitecture": [1,4,1],
    "BSMweight_clipping": 14, 
    
    "correction": "SHAPE", # "SHAPE", "NORM", ""
}

# list process normalization generation values from shape uncertainties generation values
for i in range(len(config_json["shape_nuisances_id"])):
    key = config_json["shape_nuisances_id"][i]
    if key in list(csec_nuisances_reference.keys()):
        config_json["csec_nuisances_data"][key] = config_json["shape_nuisances_data"][i]
        config_json["csec_nuisances_reference"][key]= config_json["shape_nuisances_reference"][i]

# check for errors in the config_json dictionary
n_dimensions = len(config_json["features"])
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

# add details about nuisance parameters to the folder name
correction_details = config_json["correction"]
if config_json["correction"]=='SHAPE':
    correction_details = str(len(config_json["shape_dictionary_list"]))+'_'
    for i in range(len(config_json["shape_nuisances_id"])):
        key = config_json["shape_nuisances_id"][i]
        if config_json["shape_nuisances_data"][i] !=0:
            correction_details += key.replace('AllData_', '').replace('_redTree', '').replace('_2018', '')
            correction_details += 'nu'+str(config_json["shape_nuisances_data"][i])+'_'
if config_json["correction"]=='NORM':
    correction_details = '_nu'+ config_json["norm_nuisances_data"]+'_'
ID =str(n_dimensions)+'D/'+correction_details+'Nbkg'+str(config_json["N_Bkg"])+'_Nsig'+str(config_json["N_Sig"]) 
ID+='_patience'+str(config_json["patience"])+'_epochs'+str(config_json["epochs"])
ID+='_arc'+str(config_json["BSMarchitecture"]).replace(', ', '_').replace('[', '').replace(']', '')+'_wclip'+str(config_json["BSMweight_clipping"])

def create_config_file(config_table, OUTPUT_DIRECTORY):
    with open('%s/config.json'%(OUTPUT_DIRECTORY), 'w') as outfile:
        json.dump(config_table, outfile, indent=4)
    return '%s/config.json'%(OUTPUT_DIRECTORY)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
    parser.add_argument('-p','--pyscript',     type=str, help="name of python script to execute", required=True)
    parser.add_argument('-l','--local',       type=int, help='if to be run locally', required=False, default=0)
    parser.add_argument('-t', '--toys',        type=str, default = "100", help="number of toys to be processed")

    args = parser.parse_args()
    pyscript = args.pyscript
    config_json['pyscript'] = pyscript
    pyscript = pyscript.replace('.py', '')
    pyscript = pyscript.replace('_', '/')
    config_json["output_directory"] = OUTPUT_DIRECTORY+'/'+pyscript+'/'+ID
    if args.local:
        config_json["output_directory"] = './'+ID
    if not os.path.exists(config_json["output_directory"]):
        os.makedirs(config_json["output_directory"])
    json_path = create_config_file(config_json, config_json["output_directory"])

    python_script = args.pyscript
    ntoys = args.toys
    label = 'NPLM_imperfect'+str(time.time())
    os.system("mkdir %s" %label)
    if args.local:
        os.system("source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh")
        os.system("python %s/%s -j %s" %(os.getcwd(), args.pyscript, json_path))
    else:
        for i in range(int(ntoys)):        
        
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
            #script_condor.write('requirements = (OpSysAndVer =?= "SLCern6")\n') #requirements = (OpSysAndVer =?= "CentOS7")  
