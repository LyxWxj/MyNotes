"""
Kalman Filter Visualization: 1D Constant Velocity with Dual Observations

Scenario:
  - State: [position, velocity]
  - Two independent observations combined into one:
    1. GPS: measures position (noise R_pp = 4.0)
    2. Speedometer: measures velocity (noise R_vv = 1.0)
  - Goal: fuse both observations to estimate position and velocity

Run: .venv/bin/python Notes/Tutorial/kalman_filter_dual_obs.py
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

# System matrices
F = np.array([[1, dt],        # state transition
              [0, 1]])
Q = np.array([[0.1, 0],       # process noise covariance
              [0, 0.1]])

# Combined observation model: GPS (position) + Speedometer (velocity)
H = np.array([[1, 0],          # GPS observes position
              [0, 1]])         # Speedometer observes velocity
R = np.array([[4.0, 0],        # GPS noise variance
              [0, 1.0]])       # Speedometer noise variance

# ============================================================
# 2. Generate true trajectory and observations
# ============================================================

x_true = np.zeros((N, 2))
x_true[0] = [p_true_0, v_true]

for k in range(1, N):
    w = np.random.multivariate_normal([0, 0], Q)
    x_true[k] = F @ x_true[k-1] + w

# Combined observations: z = [z_gps, z_speedometer]^T
z = np.zeros((N, 2))
for k in range(N):
    z[k] = H @ x_true[k] + np.random.multivariate_normal([0, 0], R)

# ============================================================
# 3. Kalman Filter
# ============================================================

x_est = np.zeros((N, 2))
P_est = np.zeros((N, 2, 2))
P_pred_hist = np.zeros((N, 2, 2))
K_hist = np.zeros((N, 2, 2))

x_est[0] = [0, 0]
P_est[0] = np.eye(2)

for k in range(1, N):
    # Predict
    x_pred = F @ x_est[k-1]
    P_pred = F @ P_est[k-1] @ F.T + Q
    P_pred_hist[k] = P_pred

    # Update (same formula as single observation, just bigger matrices)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    K_hist[k] = K

    x_est[k] = x_pred + (K @ (z[k] - H @ x_pred)).flatten()
    P_est[k] = (np.eye(2) - K @ H) @ P_pred

# ============================================================
# 4. Visualization
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
t = np.arange(N) * dt

# --- Subplot 1: Position ---
ax1 = axes[0]
ax1.plot(t, x_true[:, 0], 'k-', linewidth=2, label='True Position')
ax1.plot(t, x_est[:, 0], 'b-', linewidth=1.5, label='Posterior Estimate')
ax1.scatter(t, z[:, 0], s=15, c='orange', alpha=0.6, label='GPS Observations', zorder=5)

std_pos = np.sqrt(P_est[:, 0, 0])
ax1.fill_between(t,
                 x_est[:, 0] - 2*std_pos,
                 x_est[:, 0] + 2*std_pos,
                 alpha=0.2, color='blue', label='95% Confidence')

ax1.set_ylabel('Position (m)', fontsize=12)
ax1.set_title('Kalman Filter: Dual Observations (GPS + Speedometer)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Velocity ---
ax2 = axes[1]
ax2.plot(t, x_true[:, 1], 'k-', linewidth=2, label='True Velocity')
ax2.plot(t, x_est[:, 1], 'r-', linewidth=1.5, label='Posterior Estimate')
ax2.scatter(t, z[:, 1], s=15, c='green', alpha=0.6, label='Speedometer Observations', zorder=5)

std_vel = np.sqrt(P_est[:, 1, 1])
ax2.fill_between(t,
                 x_est[:, 1] - 2*std_vel,
                 x_est[:, 1] + 2*std_vel,
                 alpha=0.2, color='red', label='95% Confidence')

ax2.set_ylabel('Velocity (m/s)', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

# --- Subplot 3: Kalman Gain matrix elements ---
ax3 = axes[2]
ax3.plot(t, K_hist[:, 0, 0], 'b-', linewidth=1.5, label='$K_{pp}$ (GPS → Position)')
ax3.plot(t, K_hist[:, 0, 1], 'orange', linewidth=1.5, label='$K_{pv}$ (Speedo → Position)')
ax3.plot(t, K_hist[:, 1, 0], 'c-', linewidth=1.5, label='$K_{vp}$ (GPS → Velocity)')
ax3.plot(t, K_hist[:, 1, 1], 'm-', linewidth=1.5, label='$K_{vv}$ (Speedo → Velocity)')

ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylabel('Kalman Gain', fontsize=12)
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/media/lyxwxj/Data/ALLDocuments/obworkspace/obsrepo/Notes/Tutorial/kalman_filter_dual_obs.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Saved to Notes/Tutorial/kalman_filter_dual_obs.png")

# ============================================================
# 5. Print key steps
# ============================================================

print("\n" + "="*75)
print("First 5 steps (K matrix elements)")
print("="*75)
print(f"{'Step':>4} | {'K_pp':>8} | {'K_pv':>8} | {'K_vp':>8} | {'K_vv':>8} | {'sigma_p':>8} | {'sigma_v':>8}")
print("-"*75)
for k in range(1, 6):
    kpp = K_hist[k, 0, 0]
    kpv = K_hist[k, 0, 1]
    kvp = K_hist[k, 1, 0]
    kvv = K_hist[k, 1, 1]
    sp = np.sqrt(P_est[k, 0, 0])
    sv = np.sqrt(P_est[k, 1, 1])
    print(f"{k:>4} | {kpp:>8.4f} | {kpv:>8.4f} | {kvp:>8.4f} | {kvv:>8.4f} | {sp:>8.4f} | {sv:>8.4f}")

print("\nObservations:")
print("- K_pp and K_vv are large (diagonal: direct observation)")
print("- K_pv and K_vp are small but nonzero (cross: correlation in P)")
print("- Position std is stable (GPS prevents error accumulation)")
print("- Velocity std decreases (speedometer directly observes it)")
