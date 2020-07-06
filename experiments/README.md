 Filename | Desc  
----------|-------
al_baselines | BALD + Random acq. baselines 
ephemeral | Ephemeral + BALD experiments
calibration_al_interaction | mixup's effect on BALD acquisition
plmixup | Pseudo-labelling + mixup (see paper for algorithm)
temporal_bald | Custom acq: pick disagreeing pseudo-labels during SSL training
last_kl | Custom acq: Average KL-divergence w.r.t last pseudo-label distribution
warm_start | Warm start acquisition rounds using ephemeral pseudo-labelling
idx_finder | experiment to select fixed points (gen. 2 of avg_20): use regular accuracy and uncertainty

Directories listed below are no longer used (keeping it for posterity?)

 Filename | Desc  
----------|-------
vanilla_repeated_acquisition | Gen. 1 ephemeral experiments; superseded by `ephemeral`. Repeated acquisition (threshold) w/o regularisation techniques like mixup etc.
legacy | Gen. 1 experiments (not in use anymore, rubbish chute).
