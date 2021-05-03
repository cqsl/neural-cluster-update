# Unbiased Monte Carlo Cluster Updates with Autoregressive Neural Networks

Paper link: TODO

The code requires `Python >= 3.6` and `JAX >= 0.2.12`.

`train.py` trains a network, and `sample_cluster.py` generates samples using *neural cluster update with symmetries* (NCUS). `args.py` contains all the configurations.

`reproduce.sh` contains commands to reproduce the results in Fig. 2 and 3. In practice, you may run these commands in parallel on multiple GPUs, and set your output directory.
