File | Description
-----|--------------
last_kl | Picks points with highest KL divergence w.r.t to the last softmax distribution
last_kl_cont | Additional trials (also cancelled when server restarted (todo!))
temporal_bald | Picks points with highest pseudo-label disagreements
temporal_bald_cont | Additional trials (also cancelled when server restarted (todo!))
temporal_bald_full | Like temporal_bald, but using the full history
temporal_batch_bald | Like temporal_bald, but using batch bald (incomplete)
