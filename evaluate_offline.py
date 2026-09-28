"""
Egitilmis PEARL-tarzi modeli (checkpoint_latest.pt) test split task'larina
(ZonerAI / MctsAiZoning iceren eslesmeler) karsi ARTAN context boyutlariyla
(varsayilan 4/16/32/64) offline degerlendirir.

Her (task, context_size) icin:
  1. Task'in transitionlarindan rastgele context_size kadarini "context" olarak
     ayirir, encoder'a verip z (task embedding) cikarir.
  2. Kalan transitionlari (context'e dahil olmayanlar) "query" seti olarak
     kullanir: her query state'i icin argmax_a Q(state, z, a) ile TAHMIN
     EDILEN aksiyonu, veri setindeki GERCEK aksiyon (action_p1) ile
     karsilastirir (match_rate).
  3. Veri setindeki gercek reward'i (reward_p1), tahmin dogru cikan (match)
     ve yanlis cikan (mismatch) satirlar icin ayri ayri ortalar -- boylece
     "modelin secip veriyle ortustugu aksiyonlarin gercekte ne kadar odul
     getirdigi" gorulur (offline veride sadece GERCEKTEN ALINAN aksiyonun
     odulu bilindigi icin, tahmin=veri eslesmeyen satirlarda o TAHMIN edilen
     aksiyonun gercek odulu hesaplanamaz; sadece o satirin ait oldugu
     GERCEK aksiyonun odulu raporlanir -- bu bilinen bir offline-eval
     sinirlamasidir, gercek off-policy degerlendirme degildir).
  4. Rastgele context ornegi gurultuye yol actigi icin her (task, context_size)
     --repeats kez (varsayilan 5) farkli rastgele context ile tekrarlanip
     ortalanir.

Bu script SADECE 'bizim yontem' (egitilmis ContextEncoder+QNetwork) icin
calisir; baseline/karsilastirma eklenmedi (istege bagli sonraki adim).

Kullanim:
    ftg-env\\Scripts\\python.exe evaluate_offline.py
    ftg-env\\Scripts\\python.exe evaluate_offline.py --checkpoint checkpoints/checkpoint_epoch_0007.pt
"""
import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from build_pearl_dataset import DEFAULT_DATA_DIR, build_task_tensors
from train_pearl_smoke import ContextEncoder, QNetwork

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "checkpoint_latest.pt"
CONTEXT_SIZES = [4, 16, 32, 64]
N_REPEATS = 5


def load_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    encoder = ContextEncoder()
    q_net = QNetwork()
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    q_net.load_state_dict(ckpt["q_net_state_dict"])
    encoder.eval()
    q_net.eval()
    return encoder, q_net, ckpt.get("epoch"), ckpt.get("hparams", {})


@torch.no_grad()
def evaluate_task(encoder, q_net, task_buf, context_size, n_repeats):
    n = task_buf["state"].shape[0]
    if n <= context_size:
        return None

    out = []
    for _ in range(n_repeats):
        perm = torch.randperm(n)
        ctx_idx = perm[:context_size]
        qry_idx = perm[context_size:]

        c_state = task_buf["state"][ctx_idx].unsqueeze(0)
        c_action = task_buf["action"][ctx_idx].unsqueeze(0)
        c_reward = task_buf["reward"][ctx_idx].unsqueeze(0)
        c_next_state = task_buf["next_state"][ctx_idx].unsqueeze(0)
        z = encoder(c_state, c_action, c_reward, c_next_state)  # (1, z_dim)

        q_state = task_buf["state"][qry_idx]
        q_action_data = task_buf["action"][qry_idx]
        q_reward_data = task_buf["reward"][qry_idx]

        z_expanded = z.expand(q_state.shape[0], -1)
        q_all = q_net(q_state, z_expanded)          # (Q, n_actions)
        pred_action = q_all.argmax(dim=-1)           # (Q,)

        match = pred_action == q_action_data
        match_rate = match.float().mean().item()

        q_pred_val = q_all.gather(-1, pred_action.unsqueeze(-1)).squeeze(-1).mean().item()
        q_data_val = q_all.gather(-1, q_action_data.unsqueeze(-1)).squeeze(-1).mean().item()

        reward_all = q_reward_data.mean().item()
        reward_match = q_reward_data[match].mean().item() if match.any() else float("nan")
        reward_mismatch = q_reward_data[~match].mean().item() if (~match).any() else float("nan")

        out.append({
            "match_rate": match_rate, "reward_all": reward_all,
            "reward_match": reward_match, "reward_mismatch": reward_mismatch,
            "mean_q_pred": q_pred_val, "mean_q_data": q_data_val,
            "n_query": q_state.shape[0],
        })
    return out


def _mean(results, key):
    vals = [r[key] for r in results if r[key] == r[key]]  # NaN eler
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

    encoder, q_net, ckpt_epoch, hparams = load_model(args.checkpoint)
    print(f"Checkpoint: {args.checkpoint.resolve()} (epoch={ckpt_epoch})")
    print(f"hparams: {hparams}")

    combined = torch.load(args.data_dir / "combined.pt", weights_only=False)
    test_rows = [r for r in combined if r["split"] == "test"]
    task_tensors = build_task_tensors(test_rows)

    task_styles = {}
    for r in test_rows:
        task_styles.setdefault(r["pair_id"], (r["style_p1"], r["style_p2"]))

    test_task_ids = sorted(task_tensors.keys())
    print(f"Test task sayisi: {len(test_task_ids)} -> {test_task_ids}\n")

    print("=== BIZIM YONTEM: task x context_size detay ===")
    header = (f"{'ctx':>4} {'task':>4} {'style_p1':<12} {'style_p2':<14} "
              f"{'match_rate':>10} {'reward_all':>10} {'rw_match':>10} {'rw_mismatch':>11}")
    print(header)

    per_context = defaultdict(list)

    for context_size in args.context_sizes:
        for task_id in test_task_ids:
            style_p1, style_p2 = task_styles[task_id]
            results = evaluate_task(encoder, q_net, task_tensors[task_id], context_size, args.repeats)
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
