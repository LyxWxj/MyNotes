"""
Unscented Transform Visualization: Sigma Points

Shows how sigma points are:
1. Generated from a 2D Gaussian distribution
2. Transformed through a nonlinear function
3. Used to compute the transformed mean and covariance

Run: source .venv/bin/activate && python Notes/Tutorial/ukf_sigma_points.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from numpy.linalg import cholesky

# ============================================================
# 1. Parameters
# ============================================================

np.random.seed(42)

# Input distribution
mean = np.array([0., 0.])
cov = np.array([[32., 15.],
                [15., 40.]])

# Nonlinear function
def f_nonlinear(x):
    """Strongly nonlinear transformation"""
    return np.array([x[0] + x[1],
                     0.1 * x[0]**2 + x[1]**2])

# ============================================================
# 2. Sigma point generation (Van der Merwe Scaled)
# ============================================================

def generate_sigma_points(mean, cov, alpha=0.3, beta=2., kappa=0.1):
    """Generate 2n+1 sigma points using Van der Merwe's algorithm."""
    n = len(mean)
    lam = alpha**2 * (n + kappa) - n

    # Matrix square root (Cholesky decomposition)
    L = cholesky((n + lam) * cov)  # Lower triangular by default

    sigmas = np.zeros((2*n + 1, n))
    sigmas[0] = mean
    for i in range(n):
        sigmas[i + 1] = mean + L[i]
        sigmas[n + i + 1] = mean - L[i]

    # Weights
    Wm = np.full(2*n + 1, 1.0 / (2 * (n + lam)))
    Wc = np.full(2*n + 1, 1.0 / (2 * (n + lam)))
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    return sigmas, Wm, Wc

def unscented_transform(sigmas, Wm, Wc):
    """Compute mean and covariance from transformed sigma points."""
    mean = np.average(sigmas, weights=Wm, axis=0)
    diff = sigmas - mean
    cov = np.zeros((mean.shape[0], mean.shape[0]))
    for i in range(len(Wm)):
        cov += Wc[i] * np.outer(diff[i], diff[i])
    return mean, cov

def plot_covariance_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """Draw a covariance ellipse."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

# ============================================================
# 3. Generate sigma points and transform
# ============================================================

sigmas_in, Wm, Wc = generate_sigma_points(mean, cov)

# Transform through nonlinear function
sigmas_out = np.array([f_nonlinear(s) for s in sigmas_in])

# Compute transformed statistics
ut_mean, ut_cov = unscented_transform(sigmas_out, Wm, Wc)

# Monte Carlo comparison
np.random.seed(42)
mc_samples = 50000
mc_in = np.random.multivariate_normal(mean, cov, mc_samples)
mc_out = np.array([f_nonlinear(x) for x in mc_in])
mc_mean = mc_out.mean(axis=0)

# ============================================================
# 4. Visualization
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- Plot 1: Input sigma points ---
ax1 = axes[0]
plot_covariance_ellipse(ax1, mean, cov, n_std=1, alpha=0.15, color='blue', label='1σ ellipse')
plot_covariance_ellipse(ax1, mean, cov, n_std=2, alpha=0.08, color='blue', label='2σ ellipse')
ax1.scatter(sigmas_in[:, 0], sigmas_in[:, 1], c='red', s=100, zorder=5, label='Sigma points')
ax1.scatter(mean[0], mean[1], c='black', s=150, marker='x', linewidths=3, zorder=6, label='Mean')
for i, s in enumerate(sigmas_in):
    ax1.annotate(f'χ{i}', s, textcoords="offset points", xytext=(10, 5), fontsize=10)
ax1.set_xlabel('x₁', fontsize=12)
ax1.set_ylabel('x₂', fontsize=12)
ax1.set_title('Input: Sigma Points', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# --- Plot 2: Transformed sigma points ---
ax2 = axes[1]
plot_covariance_ellipse(ax2, ut_mean, ut_cov, n_std=1, alpha=0.15, color='green', label='1σ (UT)')
plot_covariance_ellipse(ax2, ut_mean, ut_cov, n_std=2, alpha=0.08, color='green', label='2σ (UT)')
ax2.scatter(mc_out[:3000, 0], mc_out[:3000, 1], c='lightgray', s=1, alpha=0.3, label='Monte Carlo')
ax2.scatter(sigmas_out[:, 0], sigmas_out[:, 1], c='red', s=100, zorder=5, label='Transformed σ points')
ax2.scatter(ut_mean[0], ut_mean[1], c='black', s=150, marker='x', linewidths=3, zorder=6, label='UT Mean')
ax2.scatter(mc_mean[0], mc_mean[1], c='blue', s=150, marker='+', linewidths=3, zorder=6, label='MC Mean')
for i, s in enumerate(sigmas_out):
    ax2.annotate(f'Y{i}', s, textcoords="offset points", xytext=(10, 5), fontsize=10)
ax2.set_xlabel('y₁', fontsize=12)
ax2.set_ylabel('y₂', fontsize=12)
ax2.set_title('Output: Transformed Sigma Points', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3)

# --- Plot 3: Transformation arrows ---
ax3 = axes[2]

# Normalize output to fit in same plot range as input
# Scale output points to match input scale for visualization
input_range = sigmas_in.max() - sigmas_in.min()
output_range = sigmas_out.max() - sigmas_out.min()
scale = input_range / output_range * 0.5
offset = np.array([35, 0])

sigmas_out_scaled = sigmas_out * scale + offset
ut_mean_scaled = ut_mean * scale + offset

# Draw input ellipse
plot_covariance_ellipse(ax3, mean, cov, n_std=1, alpha=0.1, color='blue')

# Arrows from input to scaled output
for i in range(len(sigmas_in)):
    ax3.annotate('', xy=sigmas_out_scaled[i], xytext=sigmas_in[i],
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=1))

ax3.scatter(sigmas_in[:, 0], sigmas_in[:, 1], c='red', s=80, zorder=5, label='Input σ points')
ax3.scatter(sigmas_out_scaled[:, 0], sigmas_out_scaled[:, 1], c='green', s=80, zorder=5, label='Output σ points (scaled)')
ax3.scatter(mean[0], mean[1], c='blue', s=100, marker='x', linewidths=2, zorder=6)
ax3.scatter(ut_mean_scaled[0], ut_mean_scaled[1], c='green', s=100, marker='x', linewidths=2, zorder=6)

ax3.set_xlabel('x₁ / y₁ (output scaled)', fontsize=12)
ax3.set_ylabel('x₂ / y₂ (output scaled)', fontsize=12)
ax3.set_title('Unscented Transform: Input → Output', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9, loc='lower right')
ax3.grid(True, alpha=0.3)
ax3.text(mean[0] - 2, mean[1] - 10, 'Input\nMean', ha='center', fontsize=9, color='blue')
ax3.text(ut_mean_scaled[0] + 2, ut_mean_scaled[1] + 3, 'Output\nMean', ha='left', fontsize=9, color='green')

plt.tight_layout()
plt.savefig('/media/lyxwxj/Data/ALLDocuments/obworkspace/obsrepo/Notes/Tutorial/ukf_sigma_points.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Saved to Notes/Tutorial/ukf_sigma_points.png")

# ============================================================
# 5. Print comparison
# ============================================================

print("\n" + "="*60)
print("Mean Comparison")
print("="*60)
print(f"{'':>20} {'y₁':>10} {'y₂':>10}")
print("-"*40)
print(f"{'Unscented Transform':>20} {ut_mean[0]:>10.3f} {ut_mean[1]:>10.3f}")
print(f"{'Monte Carlo (50k)':>20} {mc_mean[0]:>10.3f} {mc_mean[1]:>10.3f}")
print(f"{'Error':>20} {abs(ut_mean[0]-mc_mean[0]):>10.3f} {abs(ut_mean[1]-mc_mean[1]):>10.3f}")

print("\n" + "="*60)
print("Covariance Comparison")
print("="*60)
print("UT Covariance:")
print(ut_cov)
print("\nMC Covariance:")
mc_cov = np.cov(mc_out.T)
print(mc_cov)
print(f"\nMax element-wise error: {np.max(np.abs(ut_cov - mc_cov)):.3f}")
