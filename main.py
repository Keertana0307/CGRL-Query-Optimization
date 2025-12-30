# =========================================================
# 0. Reproducibility Setup
# =========================================================
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================================================
# Imports
# =========================================================
import networkx as nx
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# =========================================================
# 1. Environment
# =========================================================
class CausalDatabaseEnv:
    def __init__(self, num_tables=10, seed=SEED):
        self.rng = random.Random(seed)
        self.tables = [f"T{i}" for i in range(num_tables)]
        self.sizes = {t: self.rng.randint(5, 600000) for t in self.tables}
        self.causal_graph = nx.DiGraph()
        for i in range(num_tables - 1):
            for j in range(i + 1, min(i + 3, num_tables)):
                if self.rng.random() < 0.7:
                    self.causal_graph.add_edge(self.tables[i], self.tables[j])
        self.reset()

    def reset(self):
        self.remaining_tables = list(self.tables)
        self.join_path = []
        return self._get_state()

    def _get_state(self):
        return np.array([1 if t in self.remaining_tables else 0 for t in self.tables], dtype=np.float32)

    def get_optimal_action(self):
        if not self.join_path:
            return self.tables.index(self.tables[0])
        last = self.join_path[-1]
        for t in self.remaining_tables:
            if self.causal_graph.has_edge(last, t) or self.causal_graph.has_edge(t, last):
                return self.tables.index(t)
        return self.tables.index(self.remaining_tables[0])

    def step(self, action_idx):
        table = self.remaining_tables.pop(action_idx)
        self.join_path.append(table)
        cost = self.sizes[table] / 1000.0
        causal = True if len(self.join_path) == 1 else (
            self.causal_graph.has_edge(self.join_path[-2], table) or
            self.causal_graph.has_edge(table, self.join_path[-2])
        )
        cost *= 0.1 if causal else 2.0
        reward = -cost
        done = len(self.remaining_tables) == 0
        return self._get_state(), reward, done, table

# =========================================================
# 2. DQN Agent with Target Network
# =========================================================
class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    def push(self, *transition):
        self.memory.append(transition)
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(s),
            torch.LongTensor(a).unsqueeze(1),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(s2),
            torch.FloatTensor(d).unsqueeze(1)
        )
    def __len__(self):
        return len(self.memory)

# =========================================================
# 3. Train DQN (Double DQN + Target Network)
# =========================================================
def train_dqn(env, episodes=1500, gamma=0.95, lr=0.001, batch_size=64, target_update=50):
    state_dim = len(env.tables)
    action_dim = len(env.tables)
    model = DQNAgent(state_dim, action_dim)
    target_model = DQNAgent(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    memory = ReplayMemory()
    epsilon = 1.0
    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            available = list(range(len(env.remaining_tables)))
            if random.random() < epsilon:
                action = random.choice(available)
            else:
                q = model(torch.FloatTensor(state).unsqueeze(0))[0].detach().numpy()
                action = available[np.argmax([q[i] for i in available])]
            next_state, reward, done, _ = env.step(action)
            # reward scaling to reduce magnitude
            scaled_reward = reward / 1000.0
            memory.push(state, action, scaled_reward, next_state, done)
            state = next_state
            total_reward += scaled_reward

            if len(memory) >= batch_size:
                s, a, r, s2, d = memory.sample(batch_size)
                q_curr = model(s).gather(1, a)
                # Double DQN update
                next_actions = model(s2).argmax(1).unsqueeze(1)
                q_next = target_model(s2).gather(1, next_actions).detach()
                q_target = r + gamma * q_next * (1 - d)
                loss = loss_fn(q_curr, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(0.05, epsilon * 0.995)
        rewards_per_episode.append(total_reward)

        if ep % target_update == 0:
            target_model.load_state_dict(model.state_dict())

    return model, rewards_per_episode

# =========================================================
# 4. Optimal Policy Rollout
# =========================================================
def rollout_optimal_policy(env):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        optimal_idx = env.get_optimal_action()
        action = env.remaining_tables.index(env.tables[optimal_idx])
        state, reward, done, _ = env.step(action)
        total_reward += reward / 1000.0  # scale for comparison
    return total_reward

# =========================================================
# 5. Evaluate Join Selection Quality
# =========================================================
def evaluate_join_quality(env, model, episodes=100):
    y_true, y_pred = [], []
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            optimal = env.get_optimal_action()
            q = model(torch.FloatTensor(state).unsqueeze(0))[0].detach().numpy()
            available = list(range(len(env.remaining_tables)))
            chosen = available[np.argmax([q[i] for i in available])]
            y_true.append(optimal)
            y_pred.append(env.tables.index(env.remaining_tables[chosen]))
            state, _, done, _ = env.step(chosen)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return acc, p, r, f1

# =========================================================
# 6. Baseline Dataset
# =========================================================
def generate_dataset(env, episodes=500):
    X, y = [], []
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            X.append(state)
            y.append(env.get_optimal_action())
            action = random.choice(range(len(env.remaining_tables)))
            state, _, done, _ = env.step(action)
    return np.array(X), np.array(y)

# =========================================================
# 7. Run Experiments
# =========================================================
env = CausalDatabaseEnv()
dqn_model, dqn_rewards = train_dqn(env)

# Optimization quality
dqn_cumulative = sum(dqn_rewards)
dqn_avg_reward = np.mean(dqn_rewards)
optimal_cumulative = rollout_optimal_policy(env)

# Join quality
dqn_metrics = evaluate_join_quality(env, dqn_model)

# Baselines
X_train, y_train = generate_dataset(env)
X_test, y_test = generate_dataset(env, 100)
baselines = {
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=SEED),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=SEED)
}
baseline_results = {}
for name, clf in baselines.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    baseline_results[name] = precision_recall_fscore_support(y_test, preds, average="macro")[:4]

# =========================================================
# 8. Visualizations
# =========================================================
plt.figure(figsize=(9,4))
plt.plot(dqn_rewards)
plt.title("CG-RL Learning Curve (Reward Progression)")
plt.xlabel("Episode")
plt.ylabel("Scaled Reward")
plt.grid()
plt.show()

plt.figure(figsize=(6,4))
plt.bar(["DQN", "Optimal"], [dqn_cumulative, optimal_cumulative])
plt.title("Cumulative Reward: DQN vs Optimal Policy")
plt.ylabel("Scaled Cumulative Reward")
plt.grid(axis="y")
plt.show()

# Join selection quality bar chart
metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
all_results = {
    "CG-RL DQN": {
        "Accuracy": dqn_metrics[0],
        "Precision": dqn_metrics[1],
        "Recall": dqn_metrics[2],
        "F1-score": dqn_metrics[3],
    }
}
for name, (p, r, f1, _) in baseline_results.items():
    all_results[name] = {
        "Accuracy": accuracy_score(y_test, baselines[name].predict(X_test)),
        "Precision": p,
        "Recall": r,
        "F1-score": f1
    }

models = list(all_results.keys())
values = np.array([[all_results[m][metric] for metric in metrics_names] for m in models])
x = np.arange(len(metrics_names))
width = 0.18
plt.figure(figsize=(10,6))
for i, model in enumerate(models):
    plt.bar(x + i*width, values[i], width, label=model)
plt.xticks(x + width*(len(models)-1)/2, metrics_names)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Join Selection Quality: CG-RL vs Baselines")
plt.legend()
plt.grid(axis="y")
plt.show()

# =========================
# 9. Print Metrics
# =========================
print("\n=== Optimization Quality ===")
print(f"DQN Cumulative Reward   : {dqn_cumulative:.2f}")
print(f"Optimal Cumulative Reward: {optimal_cumulative:.2f}")
print(f"Average Episode Reward  : {dqn_avg_reward:.2f}")

print("\n=== Join Selection Quality (Text-Based Results) ===")
header = f"{'Model':<20}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1-score':>10}"
print(header)
print("-"*len(header))
for model, metrics in all_results.items():
    print(f"{model:<20}{metrics['Accuracy']:>10.4f}{metrics['Precision']:>12.4f}{metrics['Recall']:>10.4f}{metrics['F1-score']:>10.4f}")

print("\n=== Learning Efficiency Summary ===")
print(f"Initial Reward (Ep 1)   : {dqn_rewards[0]:.2f}")
print(f"Final Reward (Ep {len(dqn_rewards)}) : {dqn_rewards[-1]:.2f}")
print(f"Best Episode Reward    : {max(dqn_rewards):.2f}")
