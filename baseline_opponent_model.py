"""
Ikinci baseline: context/task-conditioning KULLANMAYAN, sadece state->action
tahmini yapan basit bir MLP siniflandirici. TUM train split (10 train task,
havuzlanmis, task ayrimi gozetmeden) uzerinde cross-entropy ile egitilir,
sonra AYNI 11 test task'ta, evaluate_offline.py ile birebir ayni
context_size x repeats x query-holdout metodolojisiyle match_rate/reward
raporlanir (context_size burada modelin tahminini ETKILEMEZ -- sadece query
setinin buyuklugunu/bilesimini diger iki script ile ozdes tutmak icin
kullanilir; boylece uc sonuc tablosu birebir karsilastirilabilir).

Bu "saf davranis klonlama" (behavior cloning) baseline'i: hicbir RL/CQL
kaybi, hicbir task embedding'i yok -- sadece "bu state'te tum train
botlarinin ortalama olarak en cok hangi aksiyonu sectigini" ogrenir.

Kullanim:
    ftg-env\\Scripts\\python.exe baseline_opponent_model.py
"""
import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from build_pearl_dataset import DEFAULT_DATA_DIR, build_task_tensors

STATE_DIM = 17
ACTION_DIM = 56
HIDDEN = 128
CONTEXT_SIZES = [4, 16, 32, 64]
N_REPEATS = 5


class OpponentModel(nn.Module):
    """Basit MLP siniflandirici: state -> 56 aksiyon uzerinde logit."""

    def __init__(self, state_dim=STATE_DIM, n_actions=ACTION_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state):
        return self.net(state)


def train_opponent_model(train_task_tensors: dict, epochs: int, batch_size: int, lr: float, seed: int):
    torch.manual_seed(seed)

    states = torch.cat([t["state"] for t in train_task_tensors.values()], dim=0)
    actions = torch.cat([t["action"] for t in train_task_tensors.values()], dim=0)
    n = states.shape[0]
    print(f"Opponent model egitim verisi: {n} satir (10 train task havuzlanmis)")

    model = OpponentModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        total_correct = 0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_state = states[idx]
            batch_action = actions[idx]

            logits = model(batch_state)
            loss = F.cross_entropy(logits, batch_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == batch_action).sum().item()
            n_batches += 1

        train_acc = total_correct / n
        print(f"[opponent-model epoch {epoch}] loss={total_loss / n_batches:.4f} "
              f"train_acc={train_acc:.4f}")

    model.eval()
    return model


@torch.no_grad()
def evaluate_task(model, task_buf, context_size, n_repeats):
    n = task_buf["state"].shape[0]
    if n <= context_size:
        return None

    out = []
    for _ in range(n_repeats):
        perm = torch.randperm(n)
        # context_size kadar satir sadece query setinden disarida birakiliyor
        # (diger iki script ile ayni query buyuklugu/bilesimi icin) --
        # bu model context KULLANMAZ, sadece state'e bakar.
        qry_idx = perm[context_size:]

        q_state = task_buf["state"][qry_idx]
        q_action_data = task_buf["action"][qry_idx]
        q_reward_data = task_buf["reward"][qry_idx]

        logits = model(q_state)
        pred_action = logits.argmax(dim=-1)

        match = pred_action == q_action_data
        match_rate = match.float().mean().item()

        reward_all = q_reward_data.mean().item()
        reward_match = q_reward_data[match].mean().item() if match.any() else float("nan")
        reward_mismatch = q_reward_data[~match].mean().item() if (~match).any() else float("nan")

        out.append({
            "match_rate": match_rate, "reward_all": reward_all,
            "reward_match": reward_match, "reward_mismatch": reward_mismatch,
            "n_query": q_state.shape[0],
        })
    return out


def _mean(results, key):
    vals = [r[key] for r in results if r[key] == r[key]]
    return statistics.fmean(vals) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--context-sizes", type=int, nargs="+", default=CONTEXT_SIZES)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--epochs", type=int, default=20, help="opponent model egitim epoch sayisi")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=Path,
                         default=Path(__file__).resolve().parent / "checkpoints" / "opponent_model.pt",
                         help="egitilen MLP'nin state_dict'inin kaydedilecegi dosya "
                              "(live_ai.py bunu yeniden egitmeden yuklemek icin kullanir)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    combined = torch.load(args.data_dir / "combined.pt", weights_only=False)
    train_rows = [r for r in combined if r["split"] == "train"]
    test_rows = [r for r in combined if r["split"] == "test"]

    train_task_tensors = build_task_tensors(train_rows)
    test_task_tensors = build_task_tensors(test_rows)

    print("=== OPPONENT MODEL EGITIMI (train split, context yok) ===")
    model = train_opponent_model(train_task_tensors, args.epochs, args.batch_size, args.lr, args.seed)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Opponent model kaydedildi: {args.save_path}")

    task_styles = {}
    for r in test_rows:
        task_styles.setdefault(r["pair_id"], (r["style_p1"], r["style_p2"]))

    test_task_ids = sorted(test_task_tensors.keys())
    print(f"\nTest task sayisi: {len(test_task_ids)} -> {test_task_ids}\n")

    print("=== BASELINE (opponent-model MLP): task x context_size detay ===")
    header = (f"{'ctx':>4} {'task':>4} {'style_p1':<12} {'style_p2':<14} "
              f"{'match_rate':>10} {'reward_all':>10} {'rw_match':>10} {'rw_mismatch':>11}")
    print(header)

    per_context = defaultdict(list)

    for context_size in args.context_sizes:
        for task_id in test_task_ids:
            style_p1, style_p2 = task_styles[task_id]
            results = evaluate_task(model, test_task_tensors[task_id], context_size, args.repeats)
            if results is None:
                print(f"{context_size:>4} {task_id:>4} {style_p1:<12} {style_p2:<14} veri yetersiz")
                continue

            row = {
                "match_rate": _mean(results, "match_rate"),
                "reward_all": _mean(results, "reward_all"),
                "reward_match": _mean(results, "reward_match"),
                "reward_mismatch": _mean(results, "reward_mismatch"),
            }
            print(f"{context_size:>4} {task_id:>4} {style_p1:<12} {style_p2:<14} "
                  f"{row['match_rate']:>10.3f} {row['reward_all']:>10.4f} "
                  f"{row['reward_match']:>10.4f} {row['reward_mismatch']:>11.4f}")
            per_context[context_size].append(row)

    print("\n=== OZET: context_size'a gore test task'lari ortalamasi ===")
    print(f"{'context':>7} {'n_task':>7} {'match_rate':>11} {'reward_all':>11} "
          f"{'rw_match':>10} {'rw_mismatch':>11}")
    for context_size in args.context_sizes:
        rows = per_context[context_size]
        if not rows:
            continue
        print(f"{context_size:>7} {len(rows):>7} "
              f"{_mean(rows, 'match_rate'):>11.3f} {_mean(rows, 'reward_all'):>11.4f} "
              f"{_mean(rows, 'reward_match'):>10.4f} {_mean(rows, 'reward_mismatch'):>11.4f}")

    print(f"\n(Referans: rastgele bir politika icin beklenen match_rate ~= 1/56 = "
          f"{1/56:.4f})")


if __name__ == "__main__":
    main()
