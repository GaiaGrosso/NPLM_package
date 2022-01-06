# NPLM_package
a package to implement the New Physics Learning Machine (NPLM) algorithm

## Short description:
NPLM is a strategy to detect data departures from a given reference model, with no prior bias on the nature of the new physics model responsible for the discrepancy. The method employs neural networks, leveraging their virtues as flexible function approximants, but builds its foundations directly on the canonical likelihood-ratio approach to hypothesis testing. The algorithm compares observations with an auxiliary set of reference-distributed events, possibly obtained with a Monte Carlo event generator. It returns a p-value, which measures the compatibility of the reference model with the data. It also identifies the most discrepant phase-space region of the dataset, to be selected for further investigation. Imperfections due to mis-modelling in the reference dataset can be taken into account straightforwardly as nuisance parameters.

## Related works:
- *"Learning New Physics from a Machine"* ([Phys. Rev. D](https://doi.org/10.1103/PhysRevD.99.015014))
- *"Learning Multivariate New Physics"* ([Eur. Phys. J. C](https://doi.org/10.1140/epjc/s10052-021-08853-y))
- *"Learning New Physics from an Imperfect Machine"* ([arXiv](https://arxiv.org/abs/2111.13633))

## Envirnoment set up:
Create a virtual environment with the packages specified in `requirements.txt`
  ```
  python3 -m venv env
  source env/bin/activate
  ```
  to be sure that pip is up to date
  ```
  pip install --upgrade pip
  ```
  install the packaes listed in `requirements.txt`
  ```
  pip install -r requirements.txt 
  ```
  to see what you installed (check if successful)
  ```
  pip freeze
  ```
  Now you are ready to download the [NPLM](https://test.pypi.org/project/NPLM/) package:
  ```
  pip install -i https://test.pypi.org/simple/ NPLM
  ```
## Example: 1D toy model
To understand how NPLM works see the 1D example in `example_1D`
