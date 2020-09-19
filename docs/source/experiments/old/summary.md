 Filename | Desc  
----------|-------
al_baselines | BALD + Random acq. baselines 
ephemeral | Ephemeral + BALD experiments
mixup | mixup's effect on BALD/BatchBALD acquisitions
plmixup | Pseudo-labelling + mixup (see paper for algorithm)
warm_start | Warm start acquisition rounds using ephemeral pseudo-labelling
idx_finder | Experiment to select fixed points (gen. 2 of avg_20): use regular accuracy and uncertainty
custom | Custom acquisition functions (mostly SSL-related)
exploration | Debugging CIFAR10's BALD performance, OOD, imbalanced, etc. (also includes CINIC10 experiments)
model_selection | Using information from `exploration`, benchmark models and choose one with the best uncertainty estimates
pure_supervised | CIFAR10 13-CNN accuracy on full dataset

Directories listed below are no longer used (keeping it for posterity?)

 Filename | Desc  
----------|-------
vanilla_repeated_acquisition | Gen. 1 ephemeral experiments; superseded by `ephemeral`. Repeated acquisition (threshold) w/o regularisation techniques like mixup etc.
legacy | Gen. 1 experiments (not in use anymore).
calibration_al_interaction | Gen. 1 al_baselines for CIFAR10 data (not used anymore)
