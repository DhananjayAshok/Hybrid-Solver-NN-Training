# Solver + NN Training of Neural Networks

This is our implementation of the GDSolver Algorithm, a hybrid training algorithm for neural networks that interprets the training of the final layer as an MILP problem, and in doing so tunnels through local minima. 

This code was written by [Dhananjay Ashok](https://dhananjay-ashok.webnode.com/), who is also the lead author on the paper on the same topic: Solver + NN Training of Neural Networks published at IJCAI22. The other authors of the paper are - Vineel Nagisetty, Christopher Srinivasa and Vijay Ganesh.

## Prerequisites
* Python 3.6+

Clone this repository:
``` bash
git clone https://github.com/DhananjayAshok/Hybrid-Solver-NN-Training GDSolver
cd GDSolver
```
## Setup
Install and Obtain a License for Gurobi Solver (GurobiPy)
Install required python packages (pytorch, numpy, pandas, matplotlib)


To run the MNIST Training Experiment 
``` bash
python train.py
```

To run other experiments, with different parameters, datasets and models see the various options of train.py (models and data in appropriate python files). 

The key addition of our algorithm are in the files
- gurobi_modules.py: MILPModel has mapping of MILP problem to and from NN
- algorithm.py: For the training algorithms. 



## Citation
If you use our work, please cite our paper. Will add link upon publication

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). For any commercial uses or derivatives, please contact us.
