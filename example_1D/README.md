# 1D toy model
A one dimensional exponential distribution is taken as Reference model.\
The presence of two sources of uncertainty, on the distribution scale and normalization respectively, makes the  knowledge of the Reference model *imperfect*. 

The NPLM algorithm tests the data sample for the presence of a signal which is not specified a priori. As a signal benchmark we take a peak in the tail of the exponential distribution, modeled by a Gaussian with average 6.44 and standard deviation 0.16. 

A fully detailed description of this use case can be found in *Learning New Physics from an Imperfect Machine* ([arXiv](https://arxiv.org/abs/2111.13633)).
## Content of this folder
- `1D_tutorial.ipynb`: shows interactively how to run a single experiment.
- `toy.py`: script to run a single experiment and save outputs.\
  Arguments:
  - `jsonfile` (`-j`): path to the json configuration file (`string`, required).
- `run_toy.py`: script to configure the experiment and execute it.\
  Usage example:
  ```
  python run_toy.pt -p toy.py
  ```
  Arguments:
  - `pyscript` (`-p`): name of the python script to be executed (`string`, required)
 
  The script creates an output folder where to store the experiment result and a json configuration file `config.json` of the following form:
  ```
  config_json = {
    "N_Ref"   : 200000,
    "N_Bkg"   : 2000,
    "N_Sig"   : 0,#10,                                                                                                                                         
    "output_directory": OUTPUT_DIRECTORY,
    "shape_nuisances_id":        ['scale'],
    "shape_nuisances_data":      [1],                                                                                                                     
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
  ```
  `config.json` is saved in the output folder.
  
- `run_toys.py` allows to submit several experiments on the CERN `lxplus` cluster via `HTCondor`.\
  Usage example:
  ```
    python run_toy.pt -p toy.py -t 100
  ```
  Arguments:
  - `pyscript` (`-p`): name of the python script to be executed (`string`, required).
  - `toys` (`-t`): number of experiments to be submitted (`int`, default 100).
  - `local` (`-l`): option that allow to execute locally one experiment (`int`, default 0).
- `analysis_outputs.ipynb`: shows how to collect the output of several experiments from an output folder and produce summary files out of them. Furthermore it displays the main plots that one can exploit to perform the analysis.
