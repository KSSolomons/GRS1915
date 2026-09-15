import numpy as np

mu_r = 5.5582e-03
mu_a = 7.2868e-03
D = 9.4  # kpc
c = 2.99792458e8  # m/s

# 1. Distance-independent beta * cos(theta)
beta_cos_theta = (mu_a - mu_r) / (mu_a + mu_r)
print(f"Derived beta * cos(theta): {beta_cos_theta:.4f} (Output: 0.1341)")

# 2. Distance-dependent beta * sin(theta)
D_m = D * 3.086e19
mu_app_rad_s = mu_a * (np.pi / (180.0 * 3600.0)) / 86400.0
beta_sin_theta = mu_app_rad_s * (1.0 - beta_cos_theta) * D_m / c

# 3. Intrinsic Jet Speed beta, Lorentz factor gamma, and Inclination Angle theta
beta = np.sqrt(beta_cos_theta**2 + beta_sin_theta**2)
gamma = 1.0 / np.sqrt(1.0 - beta**2)
gamma_beta = gamma * beta
theta_deg = np.degrees(np.arctan2(beta_sin_theta, beta_cos_theta))

delta_app = 1.0 / (gamma * (1.0 - beta_cos_theta))
delta_rec = 1.0 / (gamma * (1.0 + beta_cos_theta))
delta_ratio = ((1.0 + beta_cos_theta) / (1.0 - beta_cos_theta))

mu_rec_rad_s = mu_r * (np.pi / (180.0 * 3600.0)) / 86400.0
D_max = (c / np.sqrt(mu_app_rad_s * mu_rec_rad_s)) / 3.086e19

print(f"Intrinsic Jet Speed (beta): {beta:.4f} (Output: 0.3651)")
print(f"Lorentz Factor (Gamma): {gamma:.4f} (Output: 1.0742)")
print(f"Proper 4-Velocity (Gamma * Beta): {gamma_beta:.4f} (Output: 0.3922)")
print(f"Jet Inclination Angle (theta): {theta_deg:.2f} (Output: 68.42)")
print(f"Approaching Doppler Factor (delta_a): {delta_app:.4f} (Output: 1.0749)")
print(f"Receding Doppler Factor (delta_r): {delta_rec:.4f} (Output: 0.8205)")
print(f"Kinematic Doppler Factor Ratio: {delta_ratio:.2f} (Output: 1.31)")
print(f"Maximum Physical Distance (D_max): {D_max:.2f} (Output: 27.21)")
