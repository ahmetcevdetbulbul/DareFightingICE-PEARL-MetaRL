"""
train_pearl_smoke.py'deki ContextEncoder/QNetwork/cql_style_loss'u kullanan,
checkpoint kaydeden ve kaldigi yerden devam edebilen tam egitim dongusu.

Neden gerekli: Colab gibi oturumlar beklenmedik sekilde kopabilir. Bu script:
  - Her epoch sonunda hafif bir "checkpoint_latest.pt" yazar (model+optimizer+
    RNG durumu+kayip gecmisi) -- boylece en kotu ihtimalle son biten epoch'tan
    devam edilir.
  - Her --checkpoint-every epoch'ta bir de "checkpoint_epoch_NNNN.pt" adinda
    kalici bir milestone snapshot'i birakir (istege bagli geri donus/analiz icin).
  - Baslarken checkpoint_latest.pt varsa OTOMATIK olarak oradan devam eder;
    --no-resume ile bu kapatilip sifirdan baslanabilir.

Epoch basina adim sayisi:
  - DataLoader tek basina turunda sadece len(train_task_ids)/meta_batch kadar
    batch verir (10 task, meta_batch=4 -> 3 adim). Bu az oldugu icin artik
    --steps-per-epoch (varsayilan 24) kadar adim atilana kadar DataLoader
    gerektikce yeniden baslatiliyor -- her yeniden baslatmada task'lardan
    YENI rastgele context ornekleri cekiliyor (ayni batch tekrar edilmiyor).

Egitim kararliligi:
  - Her optimizer.step()'ten once torch.nn.utils.clip_grad_norm_ ile gradyan
    clip'lenir (--max-grad-norm, varsayilan 1.0).
  - Ayri bir target_q_net var; her adimda (her optimizer.step()'ten sonra)
    tau (--tau, varsayilan 0.005) ile yumusak guncellenir:
    target_param = tau*param + (1-tau)*target_param. TD hedefi bu target
    network ile hesaplanir (onceki surumde ayri target network YOKTU, hedef
    q_net'in kendisiyle hesaplaniyordu -- bu artik duzeltildi).
  - Varsayilan LR onceki surumun yarisina dusuruldu (train_pearl_smoke.LR/2).
  - Erken patlama tespiti: bir epoch'un mean_loss'u bir onceki epoch'un
    --explosion-factor (varsayilan 10) katini gecerse, son durum
    checkpoint'e kaydedilip egitim UYARI ile durdurulur.

Yol yonetimi: --data-dir (combined.pt'in okunacagi klasor) ve --checkpoint-dir
(checkpoint'lerin yazilacagi klasor) ikisi de bu script'in bulundugu klasore
GORELI varsayilanlara sahiptir. Yani script'i + combined.pt'i + checkpoints/
klasorunu ayni dizin duzeninde herhangi bir makineye (yerel Windows / Colab
+ Drive mount) kopyaladiginizda, hicbir yol degistirmeden calisir.

Kullanim:
    ftg-env\\Scripts\\python.exe train_pearl_full.py --epochs 50 --checkpoint-every 5
    ftg-env\\Scripts\\python.exe train_pearl_full.py --epochs 100   # ayni komut: otomatik devam eder
    # Colab ornegi (Drive mount edildikten sonra):
    python train_pearl_full.py --epochs 50 \\
        --data-dir /content/drive/MyDrive/Paper3/collected_data \\
        --checkpoint-dir /content/drive/MyDrive/Paper3/checkpoints
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from build_pearl_dataset import (DEFAULT_DATA_DIR, PearlTaskDataset, build_task_tensors,
                                  downsample_passive_runs, pearl_collate)
from train_pearl_smoke import CONTEXT_SIZE, CQL_ALPHA, GAMMA, LR, ContextEncoder, QNetwork, cql_style_loss

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_TAU = 0.005     # target network soft-update katsayisi
MAX_GRAD_NORM = 1.0     # gradyan patlamasina karsi clip esigi
EXPLOSION_FACTOR = 10.0 # mean_loss bir onceki epoch'un kac katini gecerse durdurulsun


def soft_update(target_net: torch.nn.Module, source_net: torch.nn.Module, tau: float):
    """target_param <- tau * param + (1 - tau) * target_param."""
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.mul_(1.0 - tau).add_(param, alpha=tau)


def build_train_dataset(context_size: int, data_dir: Path, downsample_passive: bool = True):
    combined_path = Path(data_dir) / "combined.pt"
    combined = torch.load(combined_path, weights_only=False)
    train_rows = [r for r in combined if r["split"] == "train"]
    if downsample_passive:
        n_before = len(train_rows)
        train_rows = downsample_passive_runs(train_rows)
        print(f"Pasif-aksiyon downsampling (STAND/BACK_STEP/CROUCH/STAND_RECOV, "
              f"5+ ardisik, %80 dusuruldu): {n_before} -> {len(train_rows)} satir "
              f"({1 - len(train_rows) / n_before:.1%} azaldi)")
    task_tensors = build_task_tensors(train_rows)
    train_task_ids = sorted(task_tensors.keys())
    return PearlTaskDataset(task_tensors, train_task_ids, context_size=context_size), train_task_ids


def save_checkpoint(path: Path, epoch: int, encoder, q_net, target_q_net, optimizer,
                     loss_history, hparams):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "q_net_state_dict": q_net.state_dict(),
        "target_q_net_state_dict": target_q_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "loss_history": loss_history,
        "hparams": hparams,
    }, path)


def load_checkpoint(path: Path, encoder, q_net, target_q_net, optimizer):
    ckpt = torch.load(path, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    q_net.load_state_dict(ckpt["q_net_state_dict"])
    if "target_q_net_state_dict" in ckpt:
        target_q_net.load_state_dict(ckpt["target_q_net_state_dict"])
    else:
        # Eski (target network eklenmeden once alinmis) checkpoint: target'i
        # q_net'in mevcut agirliklariyla baslat.
        target_q_net.load_state_dict(q_net.state_dict())
        print("UYARI: checkpoint'te target_q_net yok (eski format) -- "
              "target, q_net'in guncel agirliklariyla baslatildi.")
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    return ckpt["epoch"], ckpt.get("loss_history", []), ckpt.get("hparams", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                         help=f"combined.pt'in okunacagi klasor (varsayilan: {DEFAULT_DATA_DIR})")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
                         help=f"checkpoint'lerin yazilacagi/okunacagi klasor (varsayilan: {DEFAULT_CHECKPOINT_DIR})")
    parser.add_argument("--epochs", type=int, default=50, help="hedef toplam epoch sayisi")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="kac epoch'ta bir milestone kaydedilsin")
    parser.add_argument("--meta-batch", type=int, default=4, help="DataLoader adimi basina task sayisi")
    parser.add_argument("--steps-per-epoch", type=int, default=24,
                         help="epoch basina gradyan adimi sayisi (DataLoader, tasarlanandan az "
                              "batch verirse otomatik olarak yeniden basa sarilir; her seferinde "
                              "task'lardan YENI rastgele context ornekleri cekilir)")
    parser.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    parser.add_argument("--lr", type=float, default=LR / 2,
                         help=f"ogrenme orani (varsayilan, onceki LR={LR}'nin yarisi: {LR / 2})")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                         help="target network soft-update katsayisi")
    parser.add_argument("--cql-alpha", type=float, default=CQL_ALPHA,
                         help=f"CQL logsumexp regularizer agirligi (varsayilan {CQL_ALPHA}) -- "
                              "dusuk deger OOD aksiyonlari daha az bastirir (daha az konservatif Q-degerleri)")
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM,
                         help="gradyan clip esigi (clip_grad_norm_)")
    parser.add_argument("--explosion-factor", type=float, default=EXPLOSION_FACTOR,
                         help="mean_loss bir onceki epoch'un kac katini gecerse egitim durdurulsun")
    parser.add_argument("--no-resume", action="store_true", help="checkpoint varsa bile sifirdan basla")
    parser.add_argument("--no-passive-downsample", action="store_true",
                         help="pasif-aksiyon (STAND/BACK_STEP/CROUCH/STAND_RECOV) run-downsampling'i "
                              "kapat (varsayilan: acik -- bkz. build_pearl_dataset.downsample_passive_runs)")
    args = parser.parse_args()

    latest_ckpt = args.checkpoint_dir / "checkpoint_latest.pt"

    dataset, train_task_ids = build_train_dataset(args.context_size, args.data_dir,
                                                    downsample_passive=not args.no_passive_downsample)
    print(f"Train task sayisi: {len(train_task_ids)} -> {train_task_ids}")

    loader = DataLoader(dataset, batch_size=args.meta_batch, shuffle=True, collate_fn=pearl_collate)

    encoder = ContextEncoder()
    q_net = QNetwork()

    # Target network: q_net'in ayni mimaride, gradyanla degil sadece
    # soft-update ile guncellenen bir kopyasi. Baslangicta q_net ile birebir
    # ayni agirliklardan baslar.
    target_q_net = QNetwork()
    target_q_net.load_state_dict(q_net.state_dict())
    for p in target_q_net.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(q_net.parameters()), lr=args.lr)

    hparams = {"z_dim": 8, "context_size": args.context_size, "meta_batch": args.meta_batch,
               "steps_per_epoch": args.steps_per_epoch,
               "lr": args.lr, "gamma": GAMMA, "cql_alpha": args.cql_alpha, "tau": args.tau}

    start_epoch = 0
    loss_history = []
    if not args.no_resume and latest_ckpt.exists():
        last_epoch, loss_history, ckpt_hparams = load_checkpoint(latest_ckpt, encoder, q_net, target_q_net, optimizer)
        start_epoch = last_epoch + 1
        if ckpt_hparams and ckpt_hparams != hparams:
            raise RuntimeError(
                f"hparams uyusmazligi: checkpoint'teki ({latest_ckpt}) hparams "
                f"simdiki calisma parametreleriyle eslesmiyor.\n"
                f"  kaydedilmis={ckpt_hparams}\n"
                f"  simdiki   ={hparams}\n"
                f"Bu checkpoint'ten farkli hiperparametrelerle devam etmek modeli "
                f"tutarsiz hale getirir. --no-resume ile sifirdan baslatin ya da "
                f"komut satiri argumanlarini checkpoint'le eslesecek sekilde duzeltin."
            )
        print(f"Checkpoint bulundu ({latest_ckpt}): epoch {last_epoch}'e kadar tamamlanmis, "
              f"epoch {start_epoch}'den devam ediliyor. (gecmis={len(loss_history)} epoch)")
    else:
        print("Checkpoint yok / --no-resume: epoch 0'dan basliyor.")

    if start_epoch >= args.epochs:
        print(f"start_epoch({start_epoch}) >= hedef epochs({args.epochs}); yapilacak is kalmadi.")
        return

    all_params = list(encoder.parameters()) + list(q_net.parameters())

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        epoch_losses = []
        loader_iter = iter(loader)
        for _ in range(args.steps_per_epoch):
            try:
                batch = next(loader_iter)
            except StopIteration:
                # DataLoader'in bir turu bitti (10 task / meta-batch=4 -> 3 adim/tur).
                # Yeniden basa sar: shuffle=True oldugu icin task sirasi yeniden
                # karisir, ve PearlTaskDataset.__getitem__ her cagrida task'tan
                # YENI rastgele context ornekleri cekiyor -- yani tekrar degil,
                # gercekten farkli context'lerle ek adimlar atiliyor.
                loader_iter = iter(loader)
                batch = next(loader_iter)

            loss, td_loss, cql_reg = cql_style_loss(q_net, encoder, batch, gamma=GAMMA, cql_alpha=args.cql_alpha,
                                                      target_q_net=target_q_net)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.max_grad_norm)
            optimizer.step()
            soft_update(target_q_net, q_net, tau=args.tau)
            epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        dt = time.time() - t0
        print(f"[epoch {epoch}] mean_loss={mean_loss:.4f}  ({len(epoch_losses)} adim, {dt:.1f}s)")

        # Erken patlama tespiti: bir onceki epoch'un mean_loss'una kiyasla
        # asiri buyume varsa (loss_history henuz bos degilse) egitimi durdur.
        exploded = bool(loss_history) and mean_loss > args.explosion_factor * loss_history[-1]

        loss_history.append(mean_loss)

        # Her epoch sonunda hafif "latest" checkpoint -- oturum kopsa bile en fazla
        # 1 epoch'luk is kaybedilir. Patlama olsa da son durum kaydedilir.
        save_checkpoint(latest_ckpt, epoch, encoder, q_net, target_q_net, optimizer, loss_history, hparams)

        if exploded:
            print(f"UYARI: ERKEN PATLAMA TESPIT EDILDI! epoch {epoch} mean_loss={mean_loss:.4f}, "
                  f"bir onceki epoch'un ({loss_history[-2]:.4f}) {args.explosion_factor:.0f} katindan "
                  f"fazla. Egitim durduruluyor (son durum '{latest_ckpt}'e kaydedildi -- "
                  f"--lr dusurup ayni komutla devam edebilirsiniz).")
            return

        if (epoch + 1) % args.checkpoint_every == 0:
            milestone_path = args.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            save_checkpoint(milestone_path, epoch, encoder, q_net, target_q_net, optimizer, loss_history, hparams)
            print(f"  -> milestone checkpoint kaydedildi: {milestone_path}")

    print(f"\nEgitim tamamlandi (hedef {args.epochs} epoch, toplam gecmis {len(loss_history)} epoch). "
          f"Son checkpoint: {latest_ckpt}")


if __name__ == "__main__":
    main()
