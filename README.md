# Parallel PopGen Package Custom version

This is a custom version of Parallel PopGen Package (3P) v0.3.2 combined with the ability to ouput SFS for mutation-selection equilibirum models. The main repository can be found here: [https://github.com/DL42/ParallelPopGen](https://github.com/DL42/ParallelPopGen). It was created for the selection at synonymous sites in Drosophila paper: [https://www.biorxiv.org/content/10.1101/106476v2](https://www.biorxiv.org/content/10.1101/106476v2). The simulation programs referenced in the paper can be found in folders project/simulation and project/expectation. 

A third program project/likelihood estimates categorical DFE & pseudo-demographic parameters using maximum-likelihood for site frequency spectra in a python interface. It is a GPU-accelerated, python version of the Matlab program used in the paper [https://github.com/DL42/SFS_DFE_categorical](https://github.com/DL42/SFS_DFE_categorical). However, it was not used for the synonymous sites paper. If you wish to use it in your own work, please cite the GOFish [https://www.g3journal.org/content/7/9/3229.abstract](paper):

Accelerating Wright–Fisher Forward Simulations on the Graphics Processing Unit
David S. Lawrie
G3: GENES, GENOMES, GENETICS September 1, 2017 vol. 7 no. 9 3229-3236; https://doi.org/10.1534/g3.117.300103
