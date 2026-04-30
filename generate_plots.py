"""
Generate publication-quality visualizations for the Monte Carlo Linear Approximation GridWorld project.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('Agg')

# ──────────────────────────────────────────────
# 1. Re-implement core classes (self-contained)
# ──────────────────────────────────────────────


class GridWorld:
    def __init__(self):
        self.currentState = None
        self.actionSpace = ('U', 'D', 'L', 'R')
        self.actions = {
            (0, 0): ('D', 'R'), (0, 1): ('L', 'D', 'R'), (0, 2): ('L', 'D', 'R'), (0, 3): ('L', 'D'),
            (1, 0): ('U', 'D', 'R'), (1, 1): ('U', 'L', 'D', 'R'), (1, 2): ('U', 'L', 'D', 'R'), (1, 3): ('U', 'L', 'D'),
            (2, 0): ('U', 'D', 'R'), (2, 1): ('U', 'L', 'D', 'R'), (2, 2): ('U', 'L', 'D', 'R'), (2, 3): ('U', 'L', 'D'),
            (3, 0): ('U', 'R'), (3, 1): ('U', 'L', 'R'), (3, 2): ('U', 'L', 'R')
        }
        self.rewards = {(3, 3): 5, (1, 3): -2, (2, 1): -2, (3, 1): -2}
        self.explored = 0
        self.exploited = 0

    def getCurrentState(self):
        if not self.currentState:
            self.currentState = (0, 0)
        return self.currentState

    def is_terminal(self, s):
        return s not in self.actions

    def chooseAction(self, state, policy, exploreRate):
        if exploreRate > np.random.rand():
            self.explored += 1
            return np.random.choice(self.actions[state])
        self.exploited += 1
        return policy[state]

    def move(self, state, policy, exploreRate):
        action = self.chooseAction(state, policy, exploreRate)
        row, col = state
        if action == 'U':
            row -= 1
        elif action == 'D':
            row += 1
        elif action == 'L':
            col -= 1
        elif action == 'R':
            col += 1
        reward = self.rewards.get((row, col), 0)
        return action, (row, col), reward


class LinearApproximator:
    def __init__(self):
        self.theta = np.array([0.1, 0.1, 0.1, 0.1])
        self.theta_history = [self.theta.copy()]

    def state2Value(self, state):
        return ((state[0]-1)*self.theta[0] + (state[1]-1.5)*self.theta[1] +
                (state[0]*state[1]-3)*self.theta[2] + self.theta[3])

    def applyGD(self, state, target, learningrate=0.01):
        prediction = self.state2Value(state)
        error = target - prediction
        self.theta[0] += learningrate * error * state[0]
        self.theta[1] += learningrate * error * state[1]
        self.theta[2] += learningrate * error * (state[0] * state[1])
        self.theta[3] += learningrate * error
        self.theta_history.append(self.theta.copy())


# ──────────────────────────────────────────────
# 2. Run training with detailed logging
# ──────────────────────────────────────────────

policy = {
    (0, 0): 'R', (0, 1): 'R', (0, 2): 'D', (0, 3): 'D',
    (1, 0): 'R', (1, 1): 'D', (1, 2): 'D', (1, 3): 'D',
    (2, 0): 'R', (2, 1): 'D', (2, 2): 'R', (2, 3): 'D',
    (3, 0): 'R', (3, 1): 'R', (3, 2): 'R'
}

np.random.seed(42)
env = GridWorld()
approx = LinearApproximator()
exploreRate = 0.05
gamma = 0.9
num_episodes = 1000

episode_returns = []
episode_lengths = []
value_snapshots = []
snapshot_episodes = [0, 50, 200, 500, 999]

for ep in range(num_episodes):
    state = env.getCurrentState()
    step = 0
    trajectory = []
    while (not env.is_terminal(state)) and step < 30:
        action, nextState, reward = env.move(state, policy, exploreRate)
        trajectory.append((state, reward))
        state = nextState
        step += 1

    cumulative = 0
    for item in reversed(trajectory):
        s, r = item
        cumulative += gamma * r
        approx.applyGD(s, cumulative)

    episode_returns.append(cumulative)
    episode_lengths.append(step)

    if ep in snapshot_episodes:
        vals = {}
        for s in env.actions:
            vals[s] = approx.state2Value(s)
        vals[(3, 3)] = 5.0
        value_snapshots.append((ep, vals))

# Final values
final_values = {}
for s in env.actions:
    final_values[s] = approx.state2Value(s)
final_values[(3, 3)] = 5.0

# ──────────────────────────────────────────────
# 3. Plotting helpers
# ──────────────────────────────────────────────

STYLE = {
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
}
plt.rcParams.update(STYLE)

CMAP = LinearSegmentedColormap.from_list('custom',
                                         ['#d32f2f', '#ff9800', '#fff9c4', '#4caf50', '#1b5e20'], N=256)

action_arrows = {'U': (0, 0.3), 'D': (0, -0.3), 'L': (-0.3, 0), 'R': (0.3, 0)}


def grid_matrix(values_dict):
    mat = np.zeros((4, 4))
    for (r, c), v in values_dict.items():
        mat[r][c] = v
    return mat


# ──────────────────────────────────────────────
# PLOT 1: GridWorld Environment Diagram
# ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')
ax.invert_yaxis()
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(['Col 0', 'Col 1', 'Col 2', 'Col 3'])
ax.set_yticklabels(['Row 0', 'Row 1', 'Row 2', 'Row 3'])
ax.set_title('4×4 GridWorld Environment Layout',
             fontsize=16, fontweight='bold', pad=15)

for r in range(4):
    for c in range(4):
        color = '#e8f5e9'
        if (r, c) == (0, 0):
            color = '#bbdefb'
        elif (r, c) == (3, 3):
            color = '#c8e6c9'
        elif (r, c) in [(1, 3), (2, 1), (3, 1)]:
            color = '#ffcdd2'
        rect = plt.Rectangle((c-0.5, r-0.5), 1, 1,
                             facecolor=color, edgecolor='#37474f', linewidth=2)
        ax.add_patch(rect)

ax.text(0, 0, 'S\nStart', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#1565c0')
ax.text(3, 3, 'T\n+5', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#2e7d32')
for (r, c) in [(1, 3), (2, 1), (3, 1)]:
    ax.text(c, r, '✗\n−2', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#c62828')

for r in range(4):
    for c in range(4):
        if (r, c) not in [(0, 0), (3, 3), (1, 3), (2, 1), (3, 1)]:
            ax.text(c, r, f'({r},{c})', ha='center',
                    va='center', fontsize=9, color='#616161')

ax.grid(False)
fig.tight_layout()
fig.savefig('assets/gridworld_environment.png', dpi=200)
plt.close(fig)
print("[+] gridworld_environment.png")


# ──────────────────────────────────────────────
# PLOT 2: Policy Visualization
# ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')
ax.invert_yaxis()
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_title('Deterministic Policy π(s)',
             fontsize=16, fontweight='bold', pad=15)

for r in range(4):
    for c in range(4):
        color = '#f5f5f5'
        if (r, c) == (3, 3):
            color = '#c8e6c9'
        elif (r, c) in [(1, 3), (2, 1), (3, 1)]:
            color = '#ffcdd2'
        rect = plt.Rectangle((c-0.5, r-0.5), 1, 1,
                             facecolor=color, edgecolor='#37474f', linewidth=2)
        ax.add_patch(rect)

for state, action in policy.items():
    dx, dy = action_arrows[action]
    ax.annotate('', xy=(state[1]+dx, state[0]-dy), xytext=(state[1], state[0]),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=2.5))

ax.text(3, 3, 'GOAL\n+5', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#2e7d32')
for (r, c) in [(1, 3), (2, 1), (3, 1)]:
    ax.text(c, r, '−2', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#c62828')

ax.grid(False)
fig.tight_layout()
fig.savefig('assets/policy_visualization.png', dpi=200)
plt.close(fig)
print("[+] policy_visualization.png")


# ──────────────────────────────────────────────
# PLOT 3: Learned Value Function Heatmap
# ──────────────────────────────────────────────

mat = grid_matrix(final_values)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(mat, cmap=CMAP, interpolation='nearest')
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_title('Learned State-Value Function V(s)',
             fontsize=16, fontweight='bold', pad=15)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Estimated Value', fontsize=12)
for r in range(4):
    for c in range(4):
        val = mat[r][c]
        color = 'white' if abs(val) > (mat.max()-mat.min())*0.6 else 'black'
        ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                fontsize=13, fontweight='bold', color=color)
fig.tight_layout()
fig.savefig('assets/value_function_heatmap.png', dpi=200)
plt.close(fig)
print("[+] value_function_heatmap.png")


# ──────────────────────────────────────────────
# PLOT 4: Value Function Evolution Over Training
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle('Value Function Evolution During Training',
             fontsize=16, fontweight='bold', y=1.05)
for idx, (ep, vals) in enumerate(value_snapshots):
    mat = grid_matrix(vals)
    im = axes[idx].imshow(mat, cmap=CMAP, interpolation='nearest',
                          vmin=-3, vmax=5)
    axes[idx].set_title(f'Episode {ep+1}', fontsize=12, fontweight='bold')
    axes[idx].set_xticks(range(4))
    axes[idx].set_yticks(range(4))
    for r in range(4):
        for c in range(4):
            axes[idx].text(c, r, f'{mat[r][c]:.1f}', ha='center', va='center',
                           fontsize=9, fontweight='bold',
                           color='white' if abs(mat[r][c]) > 2 else 'black')

fig.colorbar(im, ax=axes, shrink=0.8, label='V(s)')
fig.tight_layout()
fig.savefig('assets/value_evolution.png', dpi=200)
plt.close(fig)
print("[+] value_evolution.png")


# ──────────────────────────────────────────────
# PLOT 5: Episode Returns Over Training
# ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
window = 50
smoothed = np.convolve(episode_returns, np.ones(window)/window, mode='valid')
ax.plot(episode_returns, alpha=0.2, color='#1565c0', linewidth=0.5, label='Raw')
ax.plot(range(window-1, num_episodes), smoothed, color='#1565c0',
        linewidth=2, label=f'{window}-episode moving avg')
ax.set_xlabel('Episode')
ax.set_ylabel('Cumulative Return G₀')
ax.set_title('Monte Carlo Return per Episode', fontsize=14, fontweight='bold')
ax.legend(frameon=True, fancybox=True)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('assets/training_returns.png', dpi=200)
plt.close(fig)
print("[+] training_returns.png")


# ──────────────────────────────────────────────
# PLOT 6: Parameter (θ) Convergence
# ──────────────────────────────────────────────

theta_hist = np.array(approx.theta_history)
fig, ax = plt.subplots(figsize=(10, 4))
labels = [r'$\theta_0$ (row)', r'$\theta_1$ (col)',
          r'$\theta_2$ (row×col)', r'$\theta_3$ (bias)']
colors = ['#e53935', '#1e88e5', '#43a047', '#fb8c00']
for i in range(4):
    ax.plot(theta_hist[:, i], label=labels[i], linewidth=1.5, color=colors[i])
ax.set_xlabel('Gradient Descent Update Step')
ax.set_ylabel('Parameter Value')
ax.set_title('Linear Approximator Weight Convergence θ',
             fontsize=14, fontweight='bold')
ax.legend(frameon=True, fancybox=True, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('assets/theta_convergence.png', dpi=200)
plt.close(fig)
print("[+] theta_convergence.png")


# ──────────────────────────────────────────────
# PLOT 7: Exploration vs Exploitation
# ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 5))
sizes = [env.exploited, env.explored]
labels_pie = [f'Exploited\n{env.exploited}', f'Explored\n{env.explored}']
colors_pie = ['#1565c0', '#ff8f00']
explode = (0.03, 0.06)
wedges, texts, autotexts = ax.pie(sizes, labels=labels_pie, autopct='%1.1f%%',
                                  colors=colors_pie, explode=explode,
                                  textprops={'fontsize': 11}, startangle=90,
                                  wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for t in autotexts:
    t.set_fontweight('bold')
ax.set_title(
    f'Exploration vs Exploitation (ε = {exploreRate})', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('assets/exploration_exploitation.png', dpi=200)
plt.close(fig)
print("[+] exploration_exploitation.png")


# ──────────────────────────────────────────────
# PLOT 8: Linear Approximation Surface
# ──────────────────────────────────────────────

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

rows = np.linspace(0, 3, 30)
cols = np.linspace(0, 3, 30)
R, C = np.meshgrid(rows, cols)
V = np.zeros_like(R)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        V[i, j] = approx.state2Value((R[i, j], C[i, j]))

surf = ax.plot_surface(R, C, V, cmap=CMAP, alpha=0.85, edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.6, label='V(s)')

# Scatter actual grid points
for s in env.actions:
    ax.scatter(s[0], s[1], approx.state2Value(
        s), color='black', s=50, zorder=5)
ax.scatter(3, 3, 5.0, color='#2e7d32', s=100, zorder=5, marker='*')

ax.set_xlabel('Row')
ax.set_ylabel('Column')
ax.set_zlabel('V(s)')
ax.set_title('Linear Value Function Approximation Surface',
             fontsize=14, fontweight='bold', pad=15)
ax.view_init(elev=25, azim=135)
fig.tight_layout()
fig.savefig('assets/value_surface_3d.png', dpi=200)
plt.close(fig)
print("[+] value_surface_3d.png")


# ──────────────────────────────────────────────
# PLOT 9: Tabular vs Linear Comparison
# ──────────────────────────────────────────────

# Run tabular MC for comparison
class TabularMC:
    def __init__(self):
        self.vTable = {}

    def update(self, state, g):
        if state not in self.vTable:
            self.vTable[state] = []
        self.vTable[state].append(g)

    def getValue(self, state):
        if state not in self.vTable or len(self.vTable[state]) == 0:
            return 0
        return np.mean(self.vTable[state])


np.random.seed(42)
env2 = GridWorld()
tabular = TabularMC()

for ep in range(num_episodes):
    state = env2.getCurrentState()
    step = 0
    trajectory = []
    while (not env2.is_terminal(state)) and step < 30:
        action, nextState, reward = env2.move(state, policy, exploreRate)
        trajectory.append((state, reward))
        state = nextState
        step += 1
    cumulative = 0
    for item in reversed(trajectory):
        s, r = item
        cumulative += gamma * r
        tabular.update(s, cumulative)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Tabular
tab_mat = np.zeros((4, 4))
for r in range(4):
    for c in range(4):
        if (r, c) == (3, 3):
            tab_mat[r][c] = 5.0
        else:
            tab_mat[r][c] = tabular.getValue((r, c))

im1 = axes[0].imshow(tab_mat, cmap=CMAP, interpolation='nearest')
axes[0].set_title('Tabular Monte Carlo V(s)', fontsize=13, fontweight='bold')
for r in range(4):
    for c in range(4):
        color = 'white' if abs(tab_mat[r][c]) > 2 else 'black'
        axes[0].text(c, r, f'{tab_mat[r][c]:.2f}', ha='center',
                     va='center', fontsize=12, fontweight='bold', color=color)
fig.colorbar(im1, ax=axes[0], shrink=0.8)

# Linear
lin_mat = grid_matrix(final_values)
im2 = axes[1].imshow(lin_mat, cmap=CMAP, interpolation='nearest')
axes[1].set_title('Linear Approximator V(s)', fontsize=13, fontweight='bold')
for r in range(4):
    for c in range(4):
        color = 'white' if abs(lin_mat[r][c]) > 2 else 'black'
        axes[1].text(c, r, f'{lin_mat[r][c]:.2f}', ha='center',
                     va='center', fontsize=12, fontweight='bold', color=color)
fig.colorbar(im2, ax=axes[1], shrink=0.8)

fig.suptitle('Tabular vs Linear Function Approximation',
             fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('assets/tabular_vs_linear.png', dpi=200)
plt.close(fig)
print("[+] tabular_vs_linear.png")


# ──────────────────────────────────────────────
# PLOT 10: Architecture Diagram
# ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis('off')
ax.set_title('Linear Approximation Architecture',
             fontsize=15, fontweight='bold', pad=20)

# Feature boxes
features = ['$x_0 = row$', '$x_1 = col$',
            '$x_2 = row \\times col$', '$x_3 = 1$ (bias)']
feat_y = [3.2, 2.4, 1.6, 0.8]
for i, (f, y) in enumerate(zip(features, feat_y)):
    rect = mpatches.FancyBboxPatch((0.5, y-0.25), 2.5, 0.5, boxstyle="round,pad=0.1",
                                   facecolor='#bbdefb', edgecolor='#1565c0', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(1.75, y, f, ha='center', va='center', fontsize=11)

# Theta boxes
thetas = [r'$\theta_0$', r'$\theta_1$', r'$\theta_2$', r'$\theta_3$']
for i, (t, y) in enumerate(zip(thetas, feat_y)):
    rect = mpatches.FancyBboxPatch((4.5, y-0.2), 1.2, 0.4, boxstyle="round,pad=0.1",
                                   facecolor='#fff9c4', edgecolor='#f57f17', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(5.1, y, t, ha='center', va='center', fontsize=11)
    # Arrow from feature to theta
    ax.annotate('', xy=(4.5, y), xytext=(3.0, y),
                arrowprops=dict(arrowstyle='->', color='#455a64', lw=1.5))

# Summation
circle = plt.Circle((7.5, 2), 0.4, facecolor='#c8e6c9',
                    edgecolor='#2e7d32', linewidth=2)
ax.add_patch(circle)
ax.text(7.5, 2, r'$\Sigma$', ha='center',
        va='center', fontsize=16, fontweight='bold')
for y in feat_y:
    ax.annotate('', xy=(7.1, 2), xytext=(5.7, y),
                arrowprops=dict(arrowstyle='->', color='#455a64', lw=1))

# Output
rect = mpatches.FancyBboxPatch((9, 1.7), 2.5, 0.6, boxstyle="round,pad=0.15",
                               facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
ax.add_patch(rect)
ax.text(10.25, 2, r'$\hat{V}(s) = \theta^T \phi(s)$',
        ha='center', va='center', fontsize=12, fontweight='bold')
ax.annotate('', xy=(9, 2), xytext=(7.9, 2),
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))

fig.tight_layout()
fig.savefig('assets/architecture_diagram.png', dpi=200)
plt.close(fig)
print("[+] architecture_diagram.png")

print("\n✓ All 10 plots generated successfully in assets/")
print(f"\nFinal θ = {approx.theta}")
print(f"Final V(s) values:")
for s in sorted(final_values.keys()):
    print(f"  V{s} = {final_values[s]:.4f}")
