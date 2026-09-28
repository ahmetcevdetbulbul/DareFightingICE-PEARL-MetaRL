"""
PEARL tarzi context encoder + CQL-tarzi Q-network icin minimal egitim
dongusu. build_pearl_dataset.py'nin urettigi combined.pt ve
PearlTaskDataset/pearl_collate yapisini kullanir.

Bu dosya sadece BOYUTLARIN VE KAYIPLARIN HATASIZ AKTIGINI dogrulayan bir
duman testidir: pair_id in {0, 1} kucuk alt kumesiyle 1 epoch (= 1 adim,
cunku 2 task var ve meta-batch=2) calistirir. Tam olcekli egitim (tum train
task'lari, coklu epoch, target network, learning-rate schedule vb.) icin
bu ayni sinif/kayip fonksiyonlari genisletilerek kullanilabilir.

Kullanim:
    ftg-env\\Scripts\\python.exe train_pearl_smoke.py
    ftg-env\\Scripts\\python.exe train_pearl_smoke.py --data-dir /content/drive/MyDrive/Paper3/collected_data
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from build_pearl_dataset import DEFAULT_DATA_DIR, PearlTaskDataset, build_task_tensors, pearl_collate

STATE_DIM = 17
ACTION_DIM = 56   # Action enum: NEUTRAL(0) .. STAND_D_DF_FC(55)
Z_DIM = 8
HIDDEN = 128
GAMMA = 0.99
CQL_ALPHA = 1.0
LR = 3e-4
SMOKE_PAIR_IDS = {0, 1}
CONTEXT_SIZE = 64


class ContextEncoder(nn.Module):
    """Permutasyon-invariant context encoder.

    Her (s, a, r, s') transition'i BAGIMSIZ olarak ayni MLP'den gecer,
    ardindan context ekseninde mean-pool yapilir. Mean-pool siraya duyarsiz
    oldugu icin context'teki transitionlarin sirasi z'yi degistirmez --
    yani encoder permutasyon-invaryanttir (PEARL'in bekledigi ozellik).
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, z_dim=Z_DIM, hidden=HIDDEN):
        super().__init__()
        in_dim = state_dim + action_dim + 1 + state_dim  # s, one-hot(a), r, s'
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, state, action, reward, next_state):
        # state: (K, C, state_dim) | action: (K, C) long | reward: (K, C) | next_state: (K, C, state_dim)
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        x = torch.cat([state, action_onehot, reward.unsqueeze(-1), next_state], dim=-1)
        per_transition = self.net(x)      # (K, C, z_dim) -- her transition bagimsiz encode edildi
        z = per_transition.mean(dim=1)    # (K, z_dim)    -- context uzerinde mean pooling
        return z


class QNetwork(nn.Module):
    """CQL-tarzi Q-network: girdi state+z birlesimi, cikti 56 aksiyon icin Q-degeri."""

    def __init__(self, state_dim=STATE_DIM, z_dim=Z_DIM, n_actions=ACTION_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state, z):
        # state ve z ayni onculdeki (leading) sekle broadcast/expand edilmis olmali.
        x = torch.cat([state, z], dim=-1)
        return self.net(x)  # (..., n_actions)


def cql_style_loss(q_net: QNetwork, encoder: ContextEncoder, batch: dict,
                    gamma: float = GAMMA, cql_alpha: float = CQL_ALPHA,
                    target_q_net: QNetwork = None):
    """target_q_net verilmezse (bu dosyanin duman testinde oldugu gibi) hedef
    Q, q_net'in kendisiyle hesaplanir -- basit ama ayri bir target network
    yoktur. train_pearl_full.py gibi tam olcekli egitimde ayri/yumusak
    guncellenen bir target_q_net gecirilmelidir."""
    state = batch["state"]            # (K, C, state_dim)
    action = batch["action"]          # (K, C) long
    reward = batch["reward"]          # (K, C)
    next_state = batch["next_state"]  # (K, C, state_dim)
    terminal = batch["terminal"]      # (K, C)
    K, C, _ = state.shape

    target_net = target_q_net if target_q_net is not None else q_net

    # Context'ten (bu meta-batch'in kendisinden) task embedding'i cikar.
    z = encoder(state, action, reward, next_state)             # (K, z_dim)
    z_expanded = z.unsqueeze(1).expand(K, C, z.shape[-1])       # (K, C, z_dim)

    q_all = q_net(state, z_expanded)                            # (K, C, n_actions)
    q_sa = q_all.gather(-1, action.unsqueeze(-1)).squeeze(-1)   # (K, C)

    with torch.no_grad():
        q_next_all = target_net(next_state, z_expanded)         # (K, C, n_actions)
        q_next_max = q_next_all.max(dim=-1).values              # (K, C)
        target = reward + gamma * (1.0 - terminal) * q_next_max

    td_loss = F.mse_loss(q_sa, target)

    # CQL(H) duzenlestiricisi: veri disi aksiyonlarin Q'sunu bastirir,
    # veri icindeki (gercekte alinan) aksiyonun Q'sunu yukari ceker.
    logsumexp_q = torch.logsumexp(q_all, dim=-1)  # (K, C)
    cql_reg = (logsumexp_q - q_sa).mean()

    loss = td_loss + cql_alpha * cql_reg
    return loss, td_loss, cql_reg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                         help=f"combined.pt'in bulundugu klasor (varsayilan: {DEFAULT_DATA_DIR})")
    args = parser.parse_args()

    combined_path = args.data_dir / "combined.pt"
    combined = torch.load(combined_path, weights_only=False)
    subset = [r for r in combined if r["pair_id"] in SMOKE_PAIR_IDS]
    if not subset:
        raise RuntimeError(f"pair_id {SMOKE_PAIR_IDS} icin veri bulunamadi.")

    task_tensors = build_task_tensors(subset)
    task_ids = sorted(task_tensors.keys())
    for t in task_ids:
        row0 = next(r for r in subset if r["pair_id"] == t)
        print(f"pair_id={t} ({row0['style_p1']} vs {row0['style_p2']}, split={row0['split']}): "
              f"{task_tensors[t]['state'].shape[0]} satir")

    dataset = PearlTaskDataset(task_tensors, task_ids, context_size=CONTEXT_SIZE)
    loader = DataLoader(dataset, batch_size=len(task_ids), shuffle=True, collate_fn=pearl_collate)

    encoder = ContextEncoder()
    q_net = QNetwork()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(q_net.parameters()), lr=LR)

    print(f"\nContextEncoder param sayisi: {sum(p.numel() for p in encoder.parameters())}")
    print(f"QNetwork param sayisi: {sum(p.numel() for p in q_net.parameters())}")

    print("\n=== 1 EPOCH DUMAN TESTI ===")
    for step, batch in enumerate(loader):
        print(f"[step {step}] batch shapes -> state={tuple(batch['state'].shape)} "
              f"action={tuple(batch['action'].shape)} reward={tuple(batch['reward'].shape)} "
              f"next_state={tuple(batch['next_state'].shape)} terminal={tuple(batch['terminal'].shape)} "
              f"task_id={batch['task_id'].tolist()}")

        loss, td_loss, cql_reg = cql_style_loss(q_net, encoder, batch)

        assert torch.isfinite(loss), "loss NaN/Inf!"
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[step {step}] loss={loss.item():.4f}  td_loss={td_loss.item():.4f}  "
              f"cql_reg={cql_reg.item():.4f}")

    print("\nDuman testi basarili: boyutlar ve kayip hesaplamasi hatasiz akti, "
          "backward/optimizer.step() calisti.")


if __name__ == "__main__":
    main()
