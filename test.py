import torch
from torch.distributions import Normal, Independent

# Create a batch of univariate normals (batch_shape=[2, 3], event_shape=[])
base_dist = Normal(torch.zeros(2, 3), torch.ones(2, 3))

# Reinterpret the batch dimensions [2, 3] as event dimensions
# Now, event_shape=[2, 3] and batch_shape=[]
indep_dist = Independent(base_dist, 1)

# Sample from the independent distribution (shape = event_shape = [2, 3])
sample = indep_dist.sample()

# Compute log probability (sums over all 2x3 dimensions)
log_prob = indep_dist.log_prob(sample)  # Output shape: []

print(sample.shape)
print(log_prob.shape)
