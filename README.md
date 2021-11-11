# Unbiased Monte Carlo Cluster Updates with Autoregressive Neural Networks

Paper link: [arXiv:2105.05650](https://arxiv.org/abs/2105.05650) | [Phys. Rev. Research 3, L042024 (2021)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.L042024)

The code requires Python >= 3.8 and JAX >= 0.2.12.

`train.py` trains a network, and `sample_ncus.py` generates samples using *neural cluster update with symmetries* (NCUS). `args.py` contains all the configurations.

`reproduce.sh` contains the commands to reproduce the results in Figs. 2~4. In practice, you may run these commands in parallel on multiple GPUs, and set your output directory.
