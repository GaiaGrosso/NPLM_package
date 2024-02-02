import os, json,argparse, datetime, time, glob
import numpy as np
import os.path
from config_utils import parNN_list

OUTPUT_DIRECTORY = './out/'

def create_config_file(config_table, OUTPUT_DIRECTORY):
    with open('%s/config.json'%(OUTPUT_DIRECTORY), 'w') as outfile:
        json.dump(config_table, outfile, indent=4)
    return '%s/config.json'%(OUTPUT_DIRECTORY)

# configuration dictionary
config_json = {
    "N_batches": 5,
    "N_Ref"   : int(200000/5),
    "N_Bkg"   : int(2000/5),
    "N_Sig"   : 0,#10,
    "output_directory": OUTPUT_DIRECTORY,
    "shape_nuisances_id":        ['S'],
    "shape_nuisances_data":      [0], #[1]
    "shape_nuisances_reference": [0],
    "shape_nuisances_sigma":     [0.05], 
    "shape_dictionary_list":     [parNN_list['scale']],
    "norm_nuisances_data":       0,
    "norm_nuisances_reference":  0,
    "norm_nuisances_sigma":      0.05,
    "epochs_tau": 200000,
    "patience_tau": 1000,
    "epochs_delta": 30000,
    "patience_delta": 1000,
    "BSMarchitecture": [1,4,1],
    "BSMweight_clipping": 10, 
    "correction": "", # "SHAPE", "NORM", ""
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
        #if config_json["shape_nuisances_data"][i] !=0:
        correction_details += 'sigma'+key+str(config_json["shape_nuisances_sigma"][i])+'_'+'nu'+key+str(config_json["shape_nuisances_data"][i])+'_'
if config_json["correction"]=='NORM' or config_json["correction"]=='SHAPE':
    if config_json["correction"]=='NORM':
        correction_details += '_'
    correction_details +=  'sigmaN'+ str(config_json["norm_nuisances_sigma"])+'_'+'nuN'+ str(config_json["norm_nuisances_data"])+'_'
ID ='Nbatches%i/'%(config_json["N_batches"])+correction_details+'Nbkg'+str(config_json["N_Bkg"])+'_Nsig'+str(config_json["N_Sig"]) 
ID+='_epochsTau'+str(config_json["epochs_tau"])+'_epochsDelta'+str(config_json["epochs_delta"])
ID+='_arc'+str(config_json["BSMarchitecture"]).replace(', ', '_').replace('[', '').replace(']', '')+'_wclip'+str(config_json["BSMweight_clipping"])


#### launch python script ###########################                                                                                         
if __name__ == '__main__':
    parser   = argparse.ArgumentParser()
    parser.add_argument('-p','--pyscript', type=str, help="name of python script to execute", required=True)
    parser.add_argument('-l','--local',    type=int, help='if to be run locally',             required=False, default=0)
    parser.add_argument('-t', '--toys',    type=int, help="number of toys to be processed",   required=False, default = 100)
    parser.add_argument('-s', '--firstseed', type=int, help="first seed for toys (if specified the the toys are launched with deterministic s\
eed incresing of one unit)", required=False, default=-1)
    args     = parser.parse_args()
    ntoys    = args.toys
    pyscript = args.pyscript
    firstseed= args.firstseed
    config_json['pyscript'] = pyscript

    pyscript_str = pyscript.replace('.py', '')
    pyscript_str = pyscript_str.replace('_', '/')
    config_json["output_directory"] = OUTPUT_DIRECTORY+'/'+pyscript_str+'/'+ID
    if not os.path.exists(config_json["output_directory"]):
        os.makedirs(config_json["output_directory"])

    json_path = create_config_file(config_json, config_json["output_directory"])
    if args.local:
        if firstseed<0:
            seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
            os.system("python %s/%s -j %s -s %i -r 0"%(os.getcwd(), pyscript, json_path, seed))
        else:
            os.system("python %s/%s -j %s -s %i -r 0"%(os.getcwd(), pyscript, json_path, firstseed))
    else:
        label = "folder-log-jobs"
        os.system("mkdir %s" %label)
        for i in range(ntoys):
            if firstseed>=0:
                seed=i
                seed+=firstseed
            else:
                seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
            for run in range(config_json["N_batches"]):
                script_sbatch = open("%s/submit_%i_%i.sh" %(label, seed, run) , 'w')
                script_sbatch.write("#!/bin/bash\n")
                script_sbatch.write("#SBATCH -c 1\n")
                script_sbatch.write("#SBATCH --gpus 1\n")
                script_sbatch.write("#SBATCH -t 0-12:00\n")
                script_sbatch.write("#SBATCH -p gpu\n")
                script_sbatch.write("#SBATCH --mem=4000\n")
                script_sbatch.write("#SBATCH -o %s_%i"%(pyscript_str, run)+"_%j.out\n")
                script_sbatch.write("#SBATCH -e %s_%i"%(pyscript_str, run)+"_%j.err\n")
                script_sbatch.write("\n")
                script_sbatch.write("module load python/3.10.9-fasrc01\n")
                script_sbatch.write("module load cuda/11.8.0-fasrc01\n")
                script_sbatch.write("\n")
                script_sbatch.write("python %s/%s -j %s -s %i -r %i\n"%(os.getcwd(), pyscript, json_path, seed, run))
                script_sbatch.close()
                os.system("chmod a+x %s/submit_%i_%i.sh" %(label, seed, run))
                os.system("sbatch %s/submit_%i_%i.sh"%(label, seed, run) )
