"""
evaluate_offline.py'nin ABLASYON kopyasi: z, context'ten encoder ile
hesaplanmak yerine SABIT SIFIR VEKTORU olarak kullanilir. Egitilmis
checkpoint'in AYNI q_net agirliklari kullanilir (encoder hic cagrilmaz) --
amac "context/z bilgisinin katkisi olmadan Q-network tek basina ne kadar
isabetli aksiyon seciyor" sorusuna cevap vermek.

Metodoloji, dogrudan karsilastirilabilir olmasi icin evaluate_offline.py ile
BIREBIR AYNI: her (task, context_size) icin context_size kadar satir rastgele
"context" olarak ayrilip query setinden cikarilir (burada bu satirlar z
hesaplamak icin KULLANILMAZ, sadece query seti buyuklugunu/bilesimini
evaluate_offline.py ile ozdes tutmak icin disarida birakilir), kalan satirlar
uzerinde match_rate/reward istatistikleri --repeats kez tekrarlanip ortalanir.

Kullanim:
    ftg-env\\Scripts\\python.exe baseline_no_context.py
"""
import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from build_pearl_dataset import DEFAULT_DATA_DIR, build_task_tensors
from train_pearl_smoke import Z_DIM, QNetwork

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "checkpoint_latest.pt"
CONTEXT_SIZES = [4, 16, 32, 64]
N_REPEATS = 5


def load_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    q_net = QNetwork()
    q_net.load_state_dict(ckpt["q_net_state_dict"])
    q_net.eval()
    return q_net, ckpt.get("epoch"), ckpt.get("hparams", {})


@torch.no_grad()
def evaluate_task(q_net, task_buf, context_size, n_repeats):
    n = task_buf["state"].shape[0]
    if n <= context_size:
        return None

    out = []
    for _ in range(n_repeats):
        perm = torch.randperm(n)
        # context_size kadar satir sadece query setinden disarida birakiliyor
        # (evaluate_offline.py ile ayni query buyuklugu/bilesimi icin) --
        # z hesabinda KULLANILMIYOR, z her zaman sabit sifir.
        qry_idx = perm[context_size:]

        q_state = task_buf["state"][qry_idx]
        q_action_data = task_buf["action"][qry_idx]
        q_reward_data = task_buf["reward"][qry_idx]

        z_expanded = torch.zeros(q_state.shape[0], Z_DIM)  # <-- SABIT SIFIR VEKTORU
        q_all = q_net(q_state, z_expanded)
        pred_action = q_all.argmax(dim=-1)

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
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--context-sizes", type=int, nargs="+", default=CONTEXT_SIZES)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    q_net, ckpt_epoch, hparams = load_model(args.checkpoint)
    print(f"Checkpoint: {args.checkpoint.resolve()} (epoch={ckpt_epoch})")
    print(f"hparams: {hparams}")
    print("NOT: z = sabit sifir vektoru (context/encoder KULLANILMIYOR)\n")

    combined = torch.load(args.data_dir / "combined.pt", weights_only=False)
    test_rows = [r for r in combined if r["split"] == "test"]
    task_tensors = build_task_tensors(test_rows)

    task_styles = {}
    for r in test_rows:
        task_styles.setdefault(r["pair_id"], (r["style_p1"], r["style_p2"]))

    test_task_ids = sorted(task_tensors.keys())
    print(f"Test task sayisi: {len(test_task_ids)} -> {test_task_ids}\n")

    print("=== BASELINE (z=0, context yok): task x context_size detay ===")
    header = (f"{'ctx':>4} {'task':>4} {'style_p1':<12} {'style_p2':<14} "
              f"{'match_rate':>10} {'reward_all':>10} {'rw_match':>10} {'rw_mismatch':>11}")
    print(header)

    per_context = defaultdict(list)

    for context_size in args.context_sizes:
        for task_id in test_task_ids:
            style_p1, style_p2 = task_styles[task_id]
            results = evaluate_task(q_net, task_tensors[task_id], context_size, args.repeats)
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
