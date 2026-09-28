"""
45 pairwise .pt dosyasini (10 bot, C(10,2)=45 cift) birlestirip PEARL
(Probabilistic Embeddings for Actor-critic RL) tarzi task-conditioned bir
PyTorch Dataset/DataLoader kurar.

Adimlar:
  1. collected_data/*.pt icindeki tum satirlari tek bir combined.pt'e birlestirir,
     her satira pair_id (0..44) ve split ('train'/'val'/'test') etiketi ekler.
  2. UC-YONLU task ayrimi (oncelik sirasi: test > val > train -- bir ciftte
     TEST_BOTS'tan biri varsa VAL_BOTS'tan biri de olsa dahi split='test'):
       - TEST_BOTS = {ZonerAI, MctsAiZoning}: egitimde HIC gorulmez, sadece
         nihai degerlendirmede kullanilir.
       - VAL_BOTS = {PunisherAI, AggressiveTurtle}: egitimde gorulmez,
         ilerlemeyi izlemek icin (ara degerlendirme) kullanilir.
       - Geri kalan 6 bot (RusherAI, TurtleAI, PassiveRusher, CounterPunisher,
         HybridRusherTurtle, MctsAi23i) arasindaki eslesmeler (C(6,2)=15
         pair_id) train task'i.
     Bu oncelik sirasiyla: test=17 pair, val=13 pair, train=15 pair (toplam 45).
  3. Her pair_id (=task) icin state/action/reward/next_state/terminal
     tensorlerini stack'ler (action/reward hep P1 perspektifinden, cunku
     veri toplarken agent_p1'in aksiyonu/odulu kaydedildi).
  4. PearlTaskDataset: __len__ = task sayisi, __getitem__(idx) o task'tan
     rastgele context_size kadar transition sample'lar (PEARL'in
     sample_context'ine karsilik gelir). pearl_collate_fn ile DataLoader'dan
     gelen bir meta-batch, (num_tasks, context_size, ...) sekilli tensorlere
     donusur -- yani her DataLoader adimi "K task, her birinden N ornek".

Yol yonetimi: tum girdi/cikti dosyalari TEK bir --data-dir altinda yasar
(collected_data/*.pt kaynaklari, combined.pt ve pair_meta.json ciktilari).
Varsayilan --data-dir, bu script'in bulundugu klasore GORELI olarak
belirlenir (Path(__file__).parent / "collected_data") -- yani script'i
hangi makineye tasirsaniz tasiyin (yerel Windows / Colab), dosyalari ayni
klasor duzeninde birlikte kopyaladiginiz surece hicbir yol degistirmeden
calisir.

Kullanim:
    ftg-env\\Scripts\\python.exe build_pearl_dataset.py
    ftg-env\\Scripts\\python.exe build_pearl_dataset.py --data-dir /content/drive/MyDrive/Paper3/collected_data
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "collected_data"

TEST_BOTS = {"ZonerAI", "MctsAiZoning"}
VAL_BOTS = {"PunisherAI", "AggressiveTurtle"}
CONTEXT_SIZE = 64      # PEARL sample_context boyutu: task basina kac transition
META_BATCH_SIZE = 4    # DataLoader'in bir adimda kac task ceker

# Action enum indeksleri (pyftg.models.enums.action.Action): STAND, BACK_STEP,
# CROUCH, STAND_RECOV. Train split'teki action_p1 dagiliminda bu 4 aksiyon
# tek basina >%60 pay tutuyor (frame-bazli kayit, karar-bazli degil, oldugu
# icin ardisik ayni-aksiyon dizileri asiri temsil ediliyor) -- bu da egitilen
# politikanin saldiri aksiyonlarini neredeyse hic secmemesine yol acan bir
# veri dengesizligi yaratiyor (bkz. epoch=999 canli/offline testlerdeki
# aksiyon dagilimi analizi).
PASSIVE_ACTIONS = {1, 4, 5, 16}
PASSIVE_MIN_RUN = 5
PASSIVE_DROP_RATIO = 0.8


def downsample_passive_runs(rows: list, min_run: int = PASSIVE_MIN_RUN,
                             drop_ratio: float = PASSIVE_DROP_RATIO, seed: int = 0) -> list:
    """Ardisik ayni-aksiyon 'run'lari min_run veya daha uzunsa VE o aksiyon
    PASSIVE_ACTIONS icindeyse, run'daki frame'lerin drop_ratio kadarini
    rastgele atar (her run'dan en az 1 frame korunur). rows'un frame sirasina
    gore (task icinde) siralanmis olmasi beklenir -- run'lar pair_id
    degistiginde de kesilir, boylece iki farkli task'in sinirindaki
    aksiyonlar yanlislikla tek run sayilmaz."""
    rng = random.Random(seed)
    n = len(rows)
    keep = [True] * n
    i = 0
    while i < n:
        j = i
        a = rows[i]["action_p1"]
        pid = rows[i]["pair_id"]
        while (j + 1 < n and rows[j + 1]["action_p1"] == a
               and rows[j + 1]["pair_id"] == pid):
            j += 1
        run_len = j - i + 1
        if run_len >= min_run and a in PASSIVE_ACTIONS:
            idxs = list(range(i, j + 1))
            n_drop = min(int(round(run_len * drop_ratio)), run_len - 1)
            for d in rng.sample(idxs, n_drop):
                keep[d] = False
        i = j + 1
    return [r for r, k in zip(rows, keep) if k]


def _split_for(style_p1: str, style_p2: str) -> str:
    """Oncelik sirasi: test > val > train. Bir ciftte hem test hem val botu
    varsa (ör. AggressiveTurtle vs ZonerAI) test kazanir -- test set'in
    'egitimde HIC gorulmedi' garantisi boylece bozulmaz."""
    if style_p1 in TEST_BOTS or style_p2 in TEST_BOTS:
        return "test"
    if style_p1 in VAL_BOTS or style_p2 in VAL_BOTS:
        return "val"
    return "train"


# --------------------------------------------------------------------------
# 1) Birlestirme
# --------------------------------------------------------------------------

def load_and_merge(data_dir: Path):
    """data_dir icindeki tum *.pt (combined.pt haric) dosyalarini birlestirir.
    combined.pt ve pair_meta.json'i da AYNI data_dir icine yazar."""
    data_dir = Path(data_dir)
    combined_path = data_dir / "combined.pt"
    pair_meta_path = data_dir / "pair_meta.json"

    files = sorted(f for f in data_dir.glob("*.pt") if f.name != "combined.pt")
    combined = []
    pair_meta = []

    for pair_id, f in enumerate(files):
        rows = torch.load(f, weights_only=False)
        style_p1 = rows[0]["style_p1"]
        style_p2 = rows[0]["style_p2"]
        split = _split_for(style_p1, style_p2)

        for r in rows:
            row = dict(r)
            row["pair_id"] = pair_id
            row["split"] = split
            combined.append(row)

        pair_meta.append({
            "pair_id": pair_id, "file": f.name,
            "style_p1": style_p1, "style_p2": style_p2,
            "split": split, "n_rows": len(rows),
        })

    torch.save(combined, combined_path)
    with open(pair_meta_path, "w", encoding="utf-8") as fh:
        json.dump(pair_meta, fh, indent=2, ensure_ascii=False)

    return combined, pair_meta, combined_path, pair_meta_path


# --------------------------------------------------------------------------
# 2) Task basina tensor olusturma
# --------------------------------------------------------------------------

def build_task_tensors(combined):
    by_task = defaultdict(lambda: {"state": [], "action": [], "reward": [],
                                    "next_state": [], "terminal": []})
    for r in combined:
        t = by_task[r["pair_id"]]
        t["state"].append(r["state"])
        t["action"].append(r["action_p1"])
        t["reward"].append(r["reward_p1"])
        t["next_state"].append(r["next_state"])
        t["terminal"].append(r["terminal"])

    task_tensors = {}
    for task_id, d in by_task.items():
        task_tensors[task_id] = {
            "state": torch.tensor(d["state"], dtype=torch.float32),
            "action": torch.tensor(d["action"], dtype=torch.long),
            "reward": torch.tensor(d["reward"], dtype=torch.float32),
            "next_state": torch.tensor(d["next_state"], dtype=torch.float32),
            "terminal": torch.tensor(d["terminal"], dtype=torch.float32),
        }
    return task_tensors


# --------------------------------------------------------------------------
# 3) PEARL tarzi task-conditioned Dataset
# --------------------------------------------------------------------------

class PearlTaskDataset(Dataset):
    """
    Standart PyTorch Dataset'lerden farkli olarak burada bir 'ornek' tek bir
    transition degil, TEK BIR TASK'TAN (pair_id) rastgele cekilmis
    context_size'lik bir transition demetidir. Boylece __len__ = task sayisi,
    ve DataLoader'in her adiminda K task * context_size transition gelir --
    PEARL'in "gorevi sample'la, o gorevden context cek" akisinin PyTorch
    Dataset/DataLoader karsiligi.
    """

    def __init__(self, task_tensors: dict, task_ids: list, context_size: int = CONTEXT_SIZE):
        self.task_tensors = task_tensors
        self.task_ids = list(task_ids)
        self.context_size = context_size

    def __len__(self):
        return len(self.task_ids)

    def __getitem__(self, idx):
        task_id = self.task_ids[idx]
        buf = self.task_tensors[task_id]
        n = buf["state"].shape[0]
        sample_idx = torch.randint(0, n, (self.context_size,))
        return {
            "task_id": task_id,
            "state": buf["state"][sample_idx],
            "action": buf["action"][sample_idx],
            "reward": buf["reward"][sample_idx],
            "next_state": buf["next_state"][sample_idx],
            "terminal": buf["terminal"][sample_idx],
        }


def pearl_collate(batch):
    return {
        "task_id": torch.tensor([b["task_id"] for b in batch], dtype=torch.long),
        "state": torch.stack([b["state"] for b in batch]),        # (K, context, state_dim)
        "action": torch.stack([b["action"] for b in batch]),      # (K, context)
        "reward": torch.stack([b["reward"] for b in batch]),      # (K, context)
        "next_state": torch.stack([b["next_state"] for b in batch]),
        "terminal": torch.stack([b["terminal"] for b in batch]),
    }


# --------------------------------------------------------------------------
# 4) Rapor
# --------------------------------------------------------------------------

def reward_stats(rewards: list) -> dict:
    if not rewards:
        return {"n": 0}
    return {
        "n": len(rewards),
        "mean": statistics.fmean(rewards),
        "std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "min": min(rewards),
        "max": max(rewards),
        "nonzero_ratio": sum(1 for r in rewards if r != 0.0) / len(rewards),
    }


def print_report(combined, pair_meta):
    print("\n=== SATIR SAYILARI ===")
    print(f"combined.pt toplam satir: {len(combined)}")

    rows_by_split = {s: [r for r in combined if r["split"] == s] for s in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        n = len(rows_by_split[split])
        print(f"{split:<5} satir: {n}  ({n / len(combined):.1%})")

    tasks_by_split = {s: [m for m in pair_meta if m["split"] == s] for s in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        print(f"{split:<5} task (pair) sayisi: {len(tasks_by_split[split])}")

    print("\n=== REWARD (reward_p1) ISTATISTIKLERI ===")
    for name, rows in (("TUMU", combined), ("TRAIN", rows_by_split["train"]),
                       ("VAL", rows_by_split["val"]), ("TEST", rows_by_split["test"])):
        s = reward_stats([r["reward_p1"] for r in rows])
        print(f"[{name}] n={s['n']} mean={s['mean']:.4f} std={s['std']:.4f} "
              f"min={s['min']:.1f} max={s['max']:.1f} nonzero_oran={s['nonzero_ratio']:.4f}")

    print("\n=== TASK (pair_id) BASINA OZET ===")
    print(f"{'pair_id':>7} {'split':<6} {'style_p1':<12} {'style_p2':<14} "
          f"{'n_rows':>7} {'reward_mean':>12}")
    by_task_rows = defaultdict(list)
    for r in combined:
        by_task_rows[r["pair_id"]].append(r["reward_p1"])
    for m in pair_meta:
        rs = reward_stats(by_task_rows[m["pair_id"]])
        print(f"{m['pair_id']:>7} {m['split']:<6} {m['style_p1']:<12} {m['style_p2']:<14} "
              f"{m['n_rows']:>7} {rs['mean']:>12.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                         help="collected_data/*.pt kaynaklarinin ve combined.pt/pair_meta.json "
                              "ciktilarinin bulundugu/yazilacagi klasor "
                              f"(varsayilan: {DEFAULT_DATA_DIR})")
    args = parser.parse_args()

    combined, pair_meta, combined_path, pair_meta_path = load_and_merge(args.data_dir)
    task_tensors = build_task_tensors(combined)

    task_ids_by_split = {
        split: [m["pair_id"] for m in pair_meta if m["split"] == split]
        for split in ("train", "val", "test")
    }

    datasets = {
        split: PearlTaskDataset(task_tensors, task_ids, CONTEXT_SIZE)
        for split, task_ids in task_ids_by_split.items()
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=META_BATCH_SIZE,
                             shuffle=True, collate_fn=pearl_collate),
        "val": DataLoader(datasets["val"], batch_size=META_BATCH_SIZE,
                           shuffle=False, collate_fn=pearl_collate),
        "test": DataLoader(datasets["test"], batch_size=META_BATCH_SIZE,
                            shuffle=False, collate_fn=pearl_collate),
    }

    # Duman testi: her split'ten bir meta-batch cek, sekilleri dogrula.
    print("=== DATALOADER DUMAN TESTI ===")
    for split in ("train", "val", "test"):
        batch = next(iter(loaders[split]))
        print(f"{split}_loader: len(dataset)={len(datasets[split])} task, "
              f"meta-batch state shape={tuple(batch['state'].shape)} "
              f"(K=task, context={CONTEXT_SIZE}, state_dim=17), "
              f"task_id ornegi={batch['task_id'].tolist()}")

    print_report(combined, pair_meta)

    print(f"\ncombined.pt kaydedildi: {combined_path}")
    print(f"pair_meta.json kaydedildi: {pair_meta_path}")


if __name__ == "__main__":
    main()
