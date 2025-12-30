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
    """
    Simulated multi-join query environment with causal relationships.
    Reward = negative execution cost (lower cost is better).
    """
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
        return np.array(
            [1 if t in self.remaining_tables else 0 for t in self.tables],
            dtype=np.float32
        )

    def get_optimal_action(self):
        """
        Oracle heuristic used only for:
        - supervised baselines
        - evaluation reference
        """
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

        if len(self.join_path) == 1:
            causal = True
        else:
            prev = self.join_path[-2]
            causal = (
                self.causal_graph.has_edge(prev, table)
                or self.causal_graph.has_edge(table, prev)
            )

        cost *= 0.1 if causal else 2.0
        reward = -cost
        done = len(self.remaining_tables) == 0

        return self._get_state(), reward, done, table

# =========================================================
# 2. DQN Agent
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
            torch.from_numpy(np.array(s)).float(),
            torch.LongTensor(a).unsqueeze(1),
            torch.FloatTensor(r).unsqueeze(1),
            torch.from_numpy(np.array(s2)).float(),
            torch.FloatTensor(d).unsqueeze(1)
        )

    def __len__(self):
        return len(self.memory)

# =========================================================
# 3. Train Double DQN
# =========================================================
def train_dqn(
    env,
    episodes=1500,
    gamma=0.95,
    lr=0.001,
    batch_size=64,
    target_update=50
):
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
        total_reward = 0.0

        while not done:
            available = list(range(len(env.remaining_tables)))

            if random.random() < epsilon:
                action = random.choice(available)
            else:
                q_values = model(torch.FloatTensor(state).unsqueeze(0))[0].detach().numpy()
                action = available[np.argmax([q_values[i] for i in available])]

            next_state, reward, done, _ = env.step(action)

            # Scale rewards to stabilize learning
            scaled_reward = reward / 1000.0

            memory.push(state, action, scaled_reward, next_state, done)
            state = next_state
            total_reward += scaled_reward

            if len(memory) >= batch_size:
                s, a, r, s2, d = memory.sample(batch_size)

                q_curr = model(s).gather(1, a)
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

        if ep % 200 == 0:
            print(f"Episode {ep} | Scaled Reward: {total_reward:.2f}")

    return model, rewards_per_episode

# =========================================================
# 4. Evaluation Functions
# =========================================================
def rollout_optimal_policy(env):
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        optimal_idx = env.get_optimal_action()
        action = env.remaining_tables.index(env.tables[optimal_idx])
        state, reward, done, _ = env.step(action)
        total_reward += reward / 1000.0

    return total_reward

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
# 5. Supervised Baseline Dataset (Oracle Labels)
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
# 6. Main Experiment
# =========================================================
def main():
    env = CausalDatabaseEnv()

    dqn_model, dqn_rewards = train_dqn(env)

    dqn_cumulative = sum(dqn_rewards)
    dqn_avg_reward = np.mean(dqn_rewards)
    optimal_cumulative = rollout_optimal_policy(env)

    dqn_metrics = evaluate_join_quality(env, dqn_model)

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
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
        baseline_results[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": p,
            "Recall": r,
            "F1-score": f1
        }

    # =====================================================
    # Results Output
    # =====================================================
    print("\n=== Optimization Quality (Primary Metric) ===")
    print(f"DQN Cumulative Reward    : {dqn_cumulative:.2f}")
    print(f"Optimal Policy Reward    : {optimal_cumulative:.2f}")
    print(f"Average Episode Reward   : {dqn_avg_reward:.2f}")

    print("\n=== Join Selection Quality ===")
    header = f"{'Model':<20}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>8}"
    print(header)
    print("-" * len(header))

    print(f"{'CG-RL DQN':<20}{dqn_metrics[0]:>10.4f}{dqn_metrics[1]:>12.4f}{dqn_metrics[2]:>10.4f}{dqn_metrics[3]:>8.4f}")

    for name, m in baseline_results.items():
        print(f"{name:<20}{m['Accuracy']:>10.4f}{m['Precision']:>12.4f}{m['Recall']:>10.4f}{m['F1-score']:>8.4f}")

    # =====================================================
    # Visualizations
    # =====================================================
    plt.figure(figsize=(9, 4))
    plt.plot(dqn_rewards)
    plt.title("CG-RL DQN Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Scaled Reward")
    plt.grid()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(["DQN", "Optimal"], [dqn_cumulative, optimal_cumulative])
    plt.title("Cumulative Reward Comparison (Lower Cost = Better)")
    plt.ylabel("Scaled Cumulative Reward")
    plt.grid(axis="y")
    plt.show()

# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    main()
