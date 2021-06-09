# Kalman Filter

This repository contains the implementation of the Kalman filter. Many papers either implement the kalman filter or kalman smoother and apply them when certain parameters are fixed. This repository uses the EM-algorithm to iteratively estimate the weights and provides all the derivations in [Kalman.md](Kalman.md) for a more thorough understanding. 

The following parts of the **algorithm** are implemented:
- Kalman filter
- Kalman smoother
- EM algorithm for parameter estimation

The algorithms are **tested** on
- synthetic data ([synthetic_testing.ipynb](synthetic_testing.ipynb))
- covid data ([covid.ipynb](covid.ipynb))

## Install Dependencies

The Python packages can be installed with the following command:
```
pip3 install -r requirements.txt
```


### References
