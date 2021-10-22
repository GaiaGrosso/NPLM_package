import json
import argparse
import numpy as np
import glob
import os.path
import time
from DATAutils import *
                                                            
OUTPUT_DIRECTORY = '../HZZ4L/NPLM_imperfect/4mu/SM/'                                                                 
config_json = {
    #"features": ['ZZMass'],# 'ZZPt', 'ZZEta', 'ZZPhi'],                                                                                                               
    #"features": ['Z1Mass', 'Z1Pt', 'Z1Eta', 'Z1Phi', 'Z2Mass', 'Z2Pt', 'Z2Eta', 'Z2Phi'],                                                                             
    "features": ['Z1Mass', 'Z1Pt', 'Z1Eta', 'Z2Mass', 'Z2Pt', 'Z2Eta', 'Z1Z2DeltaPhi'],
    "N_Bkg"   : 788,                                                                                                                                            
    "output_directory": OUTPUT_DIRECTORY,
    "shape_nuisances_generation": [],
    "shape_nuisances_reference": [],
    "shape_nuisances_sigma": [],
    "norm_nuisances_generation": 0,
    "norm_nuisances_reference":  0,
    "norm_nuisances_sigma":      0,
    "csec_nuisances_data":       csec_nuisances_data,
    "csec_nuisances_reference":  csec_nuisances_reference,
    "csec_nuisances_sigma":      csec_nuisances_sigma,
    "epochs": 300000,
    "patience": 10000,                                                                                                                                       
    "BSMarchitecture": [7,7,7,7,1],
    "BSMweight_clipping": 1.8,
    "correction": "", # "SHAPE", "NORM", ""                                                                                                                            
    "shape_dictionary_list": [],
}
n_dimensions = len(config_json["features"])
if config_json["BSMarchitecture"][0] != n_dimensions:
    print('Error: number of training dimensions and input layer size do not match.')
    exit()
ID =str(n_dimensions)+'D/'+config_json["correction"]+'_Nbkg'+str(config_json["N_Bkg"]) + '_patience'+str(config_json["patience"])
ID+='_epochs'+str(config_json["epochs"])+'_arc'+str(config_json["BSMarchitecture"]).replace(', ', '_').replace('[', '').replace(']', '')+'_wclip'+str(config_json["BSM\
weight_clipping"])

def create_config_file(config_table, OUTPUT_DIRECTORY):
    with open('%s/config.json'%(OUTPUT_DIRECTORY), 'w') as outfile:
        json.dump(config_table, outfile, indent=4)
    return '%s/config.json'%(OUTPUT_DIRECTORY)

	if __name__ == '__main__':
    parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface                                                                                                   
    parser.add_argument('-p','--pyscript',     type=str, help="name of python script to execute", required=True)                                                                  
    parser.add_argument('-l','--local',       type=int, help='if to be run locally', required=False, default=0)
    parser.add_argument('-t', '--toys',        type=str, default = "100", help="number of toys to be processed")
    #parser.add_argument('-j', '--json',        type=str, default = "100", help="configuration file")                                                                  
    args = parser.parse_args()
    config_json["output_directory"] = OUTPUT_DIRECTORY+ID
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
