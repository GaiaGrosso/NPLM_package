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
  `config.json` is saved in the output folder. A detailed description of each entry of the dictionary is given [below](#more-about-the-experiment-set-up).
  
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

## More about the experiment set up
Description of the parameters defined in the configuration file `config.json`:
  - `N_Ref`: number of training events for the reference sample (label=0);
  - `N_Bkg`: number of average training events for the data sample (label=1); the actual number of events will be drawn from a Poissonian distribution with expectation `N_Bkg`.
  - `N_Sig`: number of average signal events to include in the data sample; the actual number of injected signal events will be drawn from a Poissonian distribution with expectation `N_Sig`; set to 0 if it's a background-only toy experiment;                                                                                                                                         
  - `output_directory`: path to the folder where the experiment output will be saved;
  - `shape_nuisances_id`: list of labels (strings) identifying each shape effect to be included in the treatment of systematic uncertainties (example: ['scale']);
  - `shape_nuisances_data`: list of the true values of the shape nuisance parameters to be used in units of sigmas; to generate tha data sample (example: [1]);                                                                                                                     
  - `shape_nuisances_reference`: list of the values of the shape nuisance parameters that describes the reference sample in units of sigma (generally the nuisance parameters are parametrized so that they are all null values; example: [0]);
  - `shape_nuisances_sigma`: list of the values of the uncertainties associated to each shape nuisance parameter (example: [0.05]);
  - `shape_dictionary_list`: list of dictionaries containing the information regarding the parametric models associated to each shape effect; the keys list must match the `shape_nuisances_id` list (example: [parNN_list['scale']]);
  - `norm_nuisances_data`:  true value of the global normalization nuisance parameter in units of sigmas; to be used to generate the data sample (example: 0);
  - `norm_nuisances_reference`:  value of the global normalization uncertainty that describes the reference sample in units of sigmas (generally the nuisance parameters are parametrized so that they are all null values; example: 0);
  - `norm_nuisances_sigma`:  value of the uncertainty associated to the normalization nuisance parameter (example: 0.05);
  - `epochs_tau`: number of training epochs for the TAU term;
  - `patience_tau`: rate at which the training hisotry is saved for the TAU term;
  - `epochs_delta`: number of training epochs for the DELTA term;
  - `patience_delta`: ate at which the training hisotry is saved for the DELTA term;
  - `BSMarchitecture`: list of dimensions (int) characterizing the DNN architecture in the TAU term; the first number is the input dimension, the following are the number of neurons that constitute each layer, the last one is the dimension of the output layer which must be always 1 (example: [1,4,1]);
  - `BSMweight_clipping`: vlaue of the weight clipping parameter applied to each weight of the DNN in the TAU term (float);
  - `correction`: label that can take one of the following values: `"SHAPE"`, `"NORM"`, `""`; it states the training mode: if `SHAPE` both normalization and shape uncertainties are considered; if `NORM` only normalization uncertainties are considered; if `""` all systematic uncertainties are neglected and the algorithm runs a simplified version of NPLM (DELTA term is not computed and the TAU term does not contain the nuisance parameters).
