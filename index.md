---
layout: default
---

# The GDSolver Approach
## Mixed Integer Linear Program Solvers
Mixed Integer Linear Programs takes in a set of variables (real, integer or binary), a set of constraints (linear w.r.t. the variables) and (optionally) a linear objective function to maximize/ minimize. MILP Solvers are tools which can provide provably optimal solutions over the search space, or inform the user of when the input problem can never be satisfied. 

## Combining MILP Solvers with Gradient Descent
Gradient Descent is a versatile, scalable, efficient and effective way of training Neural Networks, but they perform very poorly when attempting optimization in regions with misleading gradient information such as local minima in the loss landscape. Solvers, while lacking in efficiency, do not perform gradient based search and have the ability to exhaustively search a solution space. 

The problem we turn our attention to in this work is that of creating a novel training algorithm that brings out the best feature of each training method: How can we best combine Solvers and Gradient Descent based training algorithms to produce the most efficient (in terms of scalability/versatility) and effective (in terms of loss convergence) training method? 

Our solution: The GDSolver Algorithm. 

## GDSolver Algorithm

The first step of GDSolver is to train the network with GD alone, as one would for any DNN, until the plateauing of validation loss, indicative of a local minimum. At which point, we halt the GD training and proceed to the second step- solver-based training. 

In the solver phase we take the partially trained network and focus on "Fine Tuning" the final layer using an MILP solver. GDSolver first converts the problem of training the final layer of the neural network to an MILP instance using a specialized formulation. The idea here is to search in a region around the values assigned by GD to the network's final layer weights and biases, such that the resultant assignment found by the solver has even lower loss than the one found by GD alone. If no lower loss point is found, GDSolver stops training and returns the trained DNN. 

The termination condition for the training loop is a check that ascertains whether the desired accuracy has been achieved or further improvements to the weights and biases are possible. If yes, then the loop continues, else it terminates.

For a more technical and complete explanation, please see our paper []()

## Results
In our experiments, we find that GDSolver not only scales well to additional data and very large model sizes, but also outperforms competing methods (Stochastic Gradiest Descent, Adam Optimization, Learning Rate Scheduling) in terms of rates of convergence and data efficiency. For regression tasks, GD-Solver produced models with, on average, 31.5% lower MSE in 48% less time, and for classification tasks on MNIST and CIFAR10, GDSolver was able to achieve the highest accuracy over all competing methods, using only 50% of the training data that GD baselines required.

## Citation
If you use our work, please cite our paper. [Logic Guided Genetic Algorithms](https://arxiv.org/abs/2010.11328)

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). For any commercial uses or derivatives, please contact us.

