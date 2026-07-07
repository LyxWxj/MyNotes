"""
Kalman Filter: 2D Constant Acceleration Model

State: [x, y, vx, vy, ax, ay]
  - x, y: position
  - vx, vy: velocity
  - ax, ay: acceleration

Observation: acceleration (ax, ay) with noise
  - We observe acceleration and estimate position/velocity via integration

Scenario: A vehicle starts at origin, accelerates in a curved path

Run: source .venv/bin/activate && python Notes/Tutorial/kalman_filter_2d_accel.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Parameters
# ============================================================

np.random.seed(42)

dt = 0.5           # sampling interval (s)
N = 100            # total steps

# True initial state: [x, y, vx, vy, ax, ay]
x_true_0 = np.array([0., 0., 2., 1., 0.5, 0.3])

# Process noise (acceleration noise)
q_ax = 0.5  # m/s²
q_ay = 0.5  # m/s²

# Observation noise (acceleration observation)
R = np.diag([0.5, 0.5])  # acceleration observation noise (m²/s⁴)

# ============================================================
# 2. System matrices
# ============================================================

# State transition matrix (constant acceleration model)
F = np.array([
    [1, 0, dt, 0,  0.5*dt**2, 0],
    [0, 1, 0,  dt, 0, 0.5*dt**2],
    [0, 0, 1,  0,  dt, 0],
    [0, 0, 0,  1,  0,  dt],
    [0, 0, 0,  0,  1,  0],
    [0, 0, 0,  0,  0,  1]
])

# Observation matrix (observe acceleration only)
H = np.array([
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

# Process noise covariance
# For constant acceleration model, noise enters through acceleration
G = np.array([
    [0.5*dt**2, 0],
    [0, 0.5*dt**2],
    [dt, 0],
    [0, dt],
    [1, 0],
    [0, 1]
])
Q = G @ np.diag([q_ax**2, q_ay**2]) @ G.T

# ============================================================
# 3. Generate true trajectory
# ============================================================

x_true = np.zeros((N, 6))
x_true[0] = x_true_0

for k in range(1, N):
    w = np.random.multivariate_normal([0]*6, Q)
    x_true[k] = F @ x_true[k-1] + w

# Generate observations (position only)
z = np.zeros((N, 2))
for k in range(N):
    n = np.random.multivariate_normal([0, 0], R)
    z[k] = H @ x_true[k] + n

# ============================================================
# 4. Kalman Filter
# ============================================================

x_est = np.zeros((N, 6))
P_est = np.zeros((N, 6, 6))
K_hist = np.zeros((N, 6, 2))

# Initial state estimate
x_est[0] = [0, 0, 0, 0, 0, 0]
P_est[0] = np.diag([100, 100, 10, 10, 5, 5])

for k in range(1, N):
    # Predict
    x_pred = F @ x_est[k-1]
    P_pred = F @ P_est[k-1] @ F.T + Q

    # Update
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    K_hist[k] = K

    residual = z[k] - H @ x_pred
    x_est[k] = x_pred + (K @ residual)
    P_est[k] = (np.eye(6) - K @ H) @ P_pred

# ============================================================
# 5. Visualization
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
t = np.arange(N) * dt

# --- Plot 1: 2D Trajectory ---
ax1 = axes[0, 0]
ax1.plot(x_true[:, 0], x_true[:, 1], 'k-', linewidth=2, label='True')
ax1.plot(x_est[:, 0], x_est[:, 1], 'b-', linewidth=1.5, label='Estimated')
ax1.set_xlabel('x (m)', fontsize=11)
ax1.set_ylabel('y (m)', fontsize=11)
ax1.set_title('2D Trajectory (from acceleration obs)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# --- Plot 2: Position ---
ax2 = axes[0, 1]
ax2.plot(t, x_true[:, 0], 'k-', linewidth=2, label='True x')
ax2.plot(t, x_est[:, 0], 'b-', linewidth=1.5, label='Est x')
ax2.plot(t, x_true[:, 1], 'k--', linewidth=2, label='True y')
ax2.plot(t, x_est[:, 1], 'r--', linewidth=1.5, label='Est y')
std_x = np.sqrt(P_est[:, 0, 0])
std_y = np.sqrt(P_est[:, 1, 1])
ax2.fill_between(t, x_est[:, 0]-2*std_x, x_est[:, 0]+2*std_x, alpha=0.15, color='blue')
ax2.fill_between(t, x_est[:, 1]-2*std_y, x_est[:, 1]+2*std_y, alpha=0.15, color='red')
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Position (m)', fontsize=11)
ax2.set_title('Position Estimation', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Plot 3: Velocity ---
ax3 = axes[0, 2]
ax3.plot(t, x_true[:, 2], 'k-', linewidth=2, label='True vx')
ax3.plot(t, x_est[:, 2], 'b-', linewidth=1.5, label='Est vx')
ax3.plot(t, x_true[:, 3], 'k--', linewidth=2, label='True vy')
ax3.plot(t, x_est[:, 3], 'r--', linewidth=1.5, label='Est vy')
std_vx = np.sqrt(P_est[:, 2, 2])
std_vy = np.sqrt(P_est[:, 3, 3])
ax3.fill_between(t, x_est[:, 2]-2*std_vx, x_est[:, 2]+2*std_vx, alpha=0.15, color='blue')
ax3.fill_between(t, x_est[:, 3]-2*std_vy, x_est[:, 3]+2*std_vy, alpha=0.15, color='red')
ax3.set_xlabel('Time (s)', fontsize=11)
ax3.set_ylabel('Velocity (m/s)', fontsize=11)
ax3.set_title('Velocity Estimation', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# --- Plot 4: Acceleration ---
ax4 = axes[1, 0]
ax4.plot(t, x_true[:, 4], 'k-', linewidth=2, label='True ax')
ax4.plot(t, x_est[:, 4], 'b-', linewidth=1.5, label='Est ax')
ax4.scatter(t, z[:, 0], s=5, c='green', alpha=0.4, label='Obs ax')
ax4.plot(t, x_true[:, 5], 'k--', linewidth=2, label='True ay')
ax4.plot(t, x_est[:, 5], 'r--', linewidth=1.5, label='Est ay')
ax4.scatter(t, z[:, 1], s=5, c='purple', alpha=0.4, label='Obs ay')
std_ax = np.sqrt(P_est[:, 4, 4])
std_ay = np.sqrt(P_est[:, 5, 5])
ax4.fill_between(t, x_est[:, 4]-2*std_ax, x_est[:, 4]+2*std_ax, alpha=0.15, color='blue')
ax4.fill_between(t, x_est[:, 5]-2*std_ay, x_est[:, 5]+2*std_ay, alpha=0.15, color='red')
ax4.set_xlabel('Time (s)', fontsize=11)
ax4.set_ylabel('Acceleration (m/s²)', fontsize=11)
ax4.set_title('Acceleration Estimation (directly observed)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# --- Plot 5: Estimation Error ---
ax5 = axes[1, 1]
err_pos = np.sqrt((x_true[:, 0] - x_est[:, 0])**2 + (x_true[:, 1] - x_est[:, 1])**2)
err_vel = np.sqrt((x_true[:, 2] - x_est[:, 2])**2 + (x_true[:, 3] - x_est[:, 3])**2)
err_acc = np.sqrt((x_true[:, 4] - x_est[:, 4])**2 + (x_true[:, 5] - x_est[:, 5])**2)
ax5.plot(t, err_pos, 'b-', linewidth=1.5, label='Position error')
ax5.plot(t, err_vel, 'r-', linewidth=1.5, label='Velocity error')
ax5.plot(t, err_acc, 'g-', linewidth=1.5, label='Acceleration error')
ax5.set_xlabel('Time (s)', fontsize=11)
ax5.set_ylabel('RMS Error', fontsize=11)
ax5.set_title('Estimation Error', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# --- Plot 6: Kalman Gain ---
ax6 = axes[1, 2]
ax6.plot(t, K_hist[:, 0, 0], 'b-', linewidth=1.5, label='K[x,ax]')
ax6.plot(t, K_hist[:, 2, 0], 'r-', linewidth=1.5, label='K[vx,ax]')
ax6.plot(t, K_hist[:, 4, 0], 'g-', linewidth=1.5, label='K[ax,ax]')
ax6.set_xlabel('Time (s)', fontsize=11)
ax6.set_ylabel('Kalman Gain', fontsize=11)
ax6.set_title('Kalman Gain (ax-observation)', fontsize=13, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/media/lyxwxj/Data/ALLDocuments/obworkspace/obsrepo/Notes/Tutorial/kalman_filter_2d_accel.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Saved to Notes/Tutorial/kalman_filter_2d_accel.png")

# ============================================================
# 6. Print statistics
# ============================================================

print("\n" + "="*60)
print("Final Estimation Error (last 20 steps average)")
print("="*60)
print(f"Position:     {err_pos[-20:].mean():.3f} m")
print(f"Velocity:     {err_vel[-20:].mean():.3f} m/s")
print(f"Acceleration: {err_acc[-20:].mean():.3f} m/s²")

print("\n" + "="*60)
print("State Transition Matrix F:")
print("="*60)
print(F)

print("\n" + "="*60)
print("Observation Matrix H:")
print("="*60)
print(H)
