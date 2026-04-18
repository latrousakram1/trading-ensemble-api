from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from src.utils import load_config, set_seed, save_json
    from src.data import load_market_data, temporal_split
    from src.features import add_features
    from src.rl_agent import DQN, ReplayBuffer
except ImportError:
    from utils import load_config, set_seed, save_json
    from data import load_market_data, temporal_split
    from features import add_features
    from rl_agent import DQN, ReplayBuffer


ACTIONS = {0: -1, 1: 0, 2: 1}
FEATURE_SUFFIXES = ("ret_", "sma_", "ema_", "rsi_", "vol_", "zscore_")
EXTRA_FEATURES = ["range_pct", "body_pct", "volume_zscore_20", "hour", "dayofweek", "month", "sentiment_score"]


def build_rl_table(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_SUFFIXES)] + [
        c for c in EXTRA_FEATURES if c in df.columns
    ]
    df = df.dropna(subset=feature_cols + ["close"]).reset_index(drop=True)
    X = df[feature_cols].astype("float32").values
    returns = df["close"].pct_change().shift(-1).fillna(0).values.astype("float32")
    mean = X.mean(axis=0).astype("float32")
    std = X.std(axis=0).astype("float32")
    std[std == 0] = 1.0
    X = (X - mean) / std
    return X, returns, feature_cols


def parse_hidden_dims(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_preset(mode: str, cfg: dict) -> dict:
    rl_cfg = cfg["rl"]
    base = {
        "tag": mode,
        "episodes": rl_cfg["episodes"],
        "batch_size": rl_cfg["batch_size"],
        "replay_capacity": rl_cfg["replay_capacity"],
        "target_update_every": rl_cfg["target_update_every"],
        "gamma": rl_cfg["gamma"],
        "epsilon_start": rl_cfg["epsilon_start"],
        "epsilon_end": rl_cfg["epsilon_end"],
        "epsilon_decay": rl_cfg["epsilon_decay"],
        "lr": 5e-4,
        "hidden_dims": [128, 64],
    }
    if mode == "advanced":
        base.update(
            {
                "episodes": max(150, rl_cfg["episodes"] * 2),
                "batch_size": max(128, rl_cfg["batch_size"] * 2),
                "replay_capacity": max(20000, rl_cfg["replay_capacity"] * 2),
                "target_update_every": max(200, rl_cfg["target_update_every"] * 2),
                "epsilon_decay": 0.997,
                "lr": 3e-4,
                "hidden_dims": [256, 128, 64],
            }
        )
    return base


def main():
    parser = argparse.ArgumentParser(description="Entrainement DQN baseline ou avance")
    parser.add_argument("--mode", choices=["baseline", "advanced"], default="baseline")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--target-update-every", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--epsilon-decay", type=float, default=None)
    parser.add_argument("--hidden-dims", type=str, default=None, help="Ex: 256,128,64")
    parser.add_argument("--artifact-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preset = build_preset(args.mode, cfg)
    params = {
        "mode": args.mode,
        "episodes": args.episodes or preset["episodes"],
        "batch_size": args.batch_size or preset["batch_size"],
        "replay_capacity": args.replay_capacity or preset["replay_capacity"],
        "target_update_every": args.target_update_every or preset["target_update_every"],
        "gamma": args.gamma or preset["gamma"],
        "epsilon_start": args.epsilon_start or preset["epsilon_start"],
        "epsilon_end": args.epsilon_end or preset["epsilon_end"],
        "epsilon_decay": args.epsilon_decay or preset["epsilon_decay"],
        "lr": args.lr or preset["lr"],
        "hidden_dims": parse_hidden_dims(args.hidden_dims) if args.hidden_dims else preset["hidden_dims"],
    }

    market_path = (
        Path(cfg["sentiment"]["aligned_output_csv"])
        if Path(cfg["sentiment"]["aligned_output_csv"]).exists()
        else Path(cfg["market"]["output_csv"])
    )
    df = load_market_data(market_path)
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    df = add_features(df)
    train_df, _, _ = temporal_split(df[df["asset"] == cfg["market"]["symbols"][0]].copy(), 0.70, 0.15)
    X, returns, feature_cols = build_rl_table(train_df)

    state_dim = X.shape[1]
    policy = DQN(state_dim=state_dim, hidden_dims=params["hidden_dims"]).to(device)
    target = DQN(state_dim=state_dim, hidden_dims=params["hidden_dims"]).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=params["lr"])
    buffer = ReplayBuffer(capacity=params["replay_capacity"])

    epsilon = params["epsilon_start"]
    losses = []
    rewards_hist = []
    cost = (cfg["backtest"]["fee_bps"] + cfg["backtest"]["slippage_bps"]) / 10000.0

    for ep in range(params["episodes"]):
        ep_reward = 0.0
        prev_pos = 0
        for t in range(len(X) - 1):
            state = X[t]
            next_state = X[t + 1]
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 3)
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                    action = int(q.argmax(dim=1).item())

            pos = ACTIONS[action]
            reward = pos * float(returns[t]) - cost * abs(pos - prev_pos)
            prev_pos = pos
            done = float(t == len(X) - 2)
            buffer.add(state, action, reward, next_state, done)
            ep_reward += reward

            if len(buffer) >= params["batch_size"]:
                states, actions, rewards, next_states, dones = buffer.sample(params["batch_size"])
                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                q_values = policy(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_actions = policy(next_states).argmax(dim=1, keepdim=True)
                    next_q = target(next_states).gather(1, next_actions).squeeze(1)
                    target_q = rewards + params["gamma"] * (1 - dones) * next_q

                loss = nn.SmoothL1Loss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))

            if t % params["target_update_every"] == 0:
                target.load_state_dict(policy.state_dict())

        epsilon = max(params["epsilon_end"], epsilon * params["epsilon_decay"])
        rewards_hist.append(ep_reward)
        print(
            f"Episode {ep + 1}/{params['episodes']} | reward={ep_reward:.4f} "
            f"| epsilon={epsilon:.4f} | mode={args.mode}"
        )

    model_name = args.model_name or f"dqn_{args.mode}.pt"
    metrics_name = args.artifact_name or f"dqn_{args.mode}_metrics.json"
    out_path = Path(cfg["paths"]["model_dir"]) / model_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "feature_cols": feature_cols,
            "training_params": params,
        },
        out_path,
    )

    summary = {
        "mode": args.mode,
        "episodes": params["episodes"],
        "batch_size": params["batch_size"],
        "replay_capacity": params["replay_capacity"],
        "target_update_every": params["target_update_every"],
        "lr": params["lr"],
        "gamma": params["gamma"],
        "hidden_dims": params["hidden_dims"],
        "final_epsilon": epsilon,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "mean_reward": float(np.mean(rewards_hist)) if rewards_hist else None,
        "last_reward": float(rewards_hist[-1]) if rewards_hist else None,
        "best_reward": float(np.max(rewards_hist)) if rewards_hist else None,
        "model_path": str(out_path),
    }
    save_json(summary, Path(cfg["paths"]["artifact_dir"]) / metrics_name)
    print(summary)


if __name__ == "__main__":
    main()
