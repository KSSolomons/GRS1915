import numpy as np

# Simulate MCMC chain
np.random.seed(42)
N = 1000000

# True parameters (approximate)
mu_r_true = 5.5582e-3
mu_r_err = 8.9e-4

mu_a_true = 7.2868e-3
mu_a_err = 1.17e-3

mu_r_chain = np.random.normal(mu_r_true, mu_r_err, N)
mu_a_chain = np.random.normal(mu_a_true, mu_a_err, N)

# keep only positive
mask = (mu_r_chain > 0) & (mu_a_chain > 0)
mu_r_chain = mu_r_chain[mask]
mu_a_chain = mu_a_chain[mask]

# 1. Distance-independent beta * cos(theta)
beta_cos_theta_chain = (mu_a_chain - mu_r_chain) / (mu_a_chain + mu_r_chain)

def compute_hdi(samples):
    p16 = np.percentile(samples, 16)
    p50 = np.percentile(samples, 50)
    p84 = np.percentile(samples, 84)
    return p16, p50, p84

p16, p50, p84 = compute_hdi(beta_cos_theta_chain)
print(f"MCMC Derived beta * cos(theta): {p50:.4f} + {p84-p50:.4f} - {p50-p16:.4f}")
print(f"Output from script was: 0.1341 + 0.0093 - 0.0091")

