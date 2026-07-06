"""
Kalman Filter Visualization: 1D Constant Velocity Motion

Scenario:
  - State: [position, velocity]
  - Observation: velocity only (noisy)
  - Goal: estimate position from velocity observations

Run: .venv/bin/python Notes/Tutorial/kalman_filter_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Parameters
# ============================================================

np.random.seed(42)

dt = 1.0          # sampling interval (s)
N = 50            # total steps
v_true = 1.0      # true velocity (m/s)
p_true_0 = 0.0    # true initial position (m)

# Noise parameters
Q = np.array([[0.1, 0],    # process noise covariance
              [0, 0.1]])
R = np.array([[1.0]])       # observation noise covariance (velocity only)

# Observation matrix: observe velocity only
H = np.array([[0, 1]])

# ============================================================
# 2. Generate true trajectory and observations
# ============================================================

# True state [position, velocity]
x_true = np.zeros((N, 2))
x_true[0] = [p_true_0, v_true]

for k in range(1, N):
    w = np.random.multivariate_normal([0, 0], Q)
    x_true[k, 0] = x_true[k-1, 0] + x_true[k-1, 1] * dt + w[0]
    x_true[k, 1] = x_true[k-1, 1] + w[1]

# Observations: velocity + noise
z = np.zeros((N, 1))
for k in range(N):
    n = np.random.multivariate_normal([0], R)
    z[k] = H @ x_true[k] + n

# ============================================================
# 3. Kalman Filter
# ============================================================

# State transition matrix
F = np.array([[1, dt],
              [0, 1]])

# Initial state
x_est = np.zeros((N, 2))       # posterior estimate
x_pred = np.zeros((N, 2))      # prior estimate
P_est = np.zeros((N, 2, 2))    # posterior covariance
P_pred = np.zeros((N, 2, 2))   # prior covariance
K_hist = np.zeros((N, 2, 1))   # Kalman gain

# Initial values
x_est[0] = [0, 0]
P_est[0] = np.array([[1, 0],
                      [0, 1]])

for k in range(1, N):
    # --- Predict ---
    x_pred[k] = F @ x_est[k-1]
    P_pred[k] = F @ P_est[k-1] @ F.T + Q

    # --- Update ---
    S = H @ P_pred[k] @ H.T + R          # innovation covariance
    K = P_pred[k] @ H.T @ np.linalg.inv(S)
    K_hist[k] = K

    residual = z[k] - H @ x_pred[k]      # innovation
    x_est[k] = x_pred[k] + (K @ residual).flatten()
    P_est[k] = (np.eye(2) - K @ H) @ P_pred[k]

# ============================================================
# 4. Visualization
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
t = np.arange(N) * dt

# --- Subplot 1: Position ---
ax1 = axes[0]
ax1.plot(t, x_true[:, 0], 'k-', linewidth=2, label='True Position')
ax1.plot(t, x_est[:, 0], 'b-', linewidth=1.5, label='Posterior Estimate')
ax1.plot(t, x_pred[:, 0], 'b--', alpha=0.5, linewidth=1, label='Prior Estimate')

std_pos = np.sqrt(P_est[:, 0, 0])
ax1.fill_between(t,
                 x_est[:, 0] - 2*std_pos,
                 x_est[:, 0] + 2*std_pos,
                 alpha=0.2, color='blue', label='95% Confidence')

ax1.set_ylabel('Position (m)', fontsize=12)
ax1.set_title('Kalman Filter: 1D Constant Velocity Motion', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Velocity ---
ax2 = axes[1]
ax2.plot(t, x_true[:, 1], 'k-', linewidth=2, label='True Velocity')
ax2.plot(t, x_est[:, 1], 'r-', linewidth=1.5, label='Posterior Estimate')
ax2.scatter(t, z.flatten(), s=15, c='green', alpha=0.6, label='Observations', zorder=5)

std_vel = np.sqrt(P_est[:, 1, 1])
ax2.fill_between(t,
                 x_est[:, 1] - 2*std_vel,
                 x_est[:, 1] + 2*std_vel,
                 alpha=0.2, color='red', label='95% Confidence')

ax2.set_ylabel('Velocity (m/s)', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

# --- Subplot 3: Kalman Gain & Std Dev ---
ax3 = axes[2]
ax3.plot(t, K_hist[:, 0, 0], 'g-', linewidth=1.5, label='Position Gain $K_p$')
ax3.plot(t, K_hist[:, 1, 0], 'm-', linewidth=1.5, label='Velocity Gain $K_v$')
ax3.plot(t, std_pos, 'b--', linewidth=1, alpha=0.7, label='Position Std $\\sigma_p$')
ax3.plot(t, std_vel, 'r--', linewidth=1, alpha=0.7, label='Velocity Std $\\sigma_v$')

ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylabel('Gain / Std Dev', fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/media/lyxwxj/Data/ALLDocuments/obworkspace/obsrepo/Notes/Tutorial/kalman_filter_demo.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Saved to Notes/Tutorial/kalman_filter_demo.png")

# ============================================================
# 5. Print key steps
# ============================================================

print("\n" + "="*65)
print("First 5 steps (position dimension)")
print("="*65)
print(f"{'Step':>4} | {'sigma1':>8} | {'sigma2':>8} | {'gain k':>8} | {'sigma':>8}")
print("-"*65)
for k in range(1, 6):
    sigma1 = P_pred[k, 0, 0]
    sigma2 = R[0, 0]
    gain = K_hist[k, 1, 0]
    sigma = P_est[k, 0, 0]
    print(f"{k:>4} | {sigma1:>8.4f} | {sigma2:>8.4f} | {gain:>8.4f} | {sigma:>8.4f}")

print("\nObservations:")
print("- sigma < sigma1: fusion reduces uncertainty")
print("- gain k stabilizes over iterations (steady state)")
print("- posterior variance converges to steady state")
