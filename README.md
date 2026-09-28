# DareFightingICE PEARL Meta-RL

[DareFightingICE 7.1](https://www.ice.ci.ritsumei.ac.jp/~ftgaic/) dövüş oyunu motoru üzerinde, `pyftg` arayüzünü kullanarak kural-tabanlı botlarla veri toplayan, PEARL-tarzı (context-conditioned, permütasyon-invariant context encoder + CQL Q-network) bir meta-RL ajanını eğiten ve hem offline hem canlı oyunda değerlendiren uçtan uca bir pipeline.

## Genel Bakış

1. **Rakip botlar** (`rule_based_bots.py`): 8 farklı kural-tabanlı Python botu (RusherAI, TurtleAI, ZonerAI, PunisherAI, PassiveRusher, CounterPunisher, AggressiveTurtle, HybridRusherTurtle) + motorun 2 yerleşik Java botu (MctsAi23i, MctsAiZoning) = 10 bot, C(10,2)=45 ikili eşleşme.
2. **Veri toplama** (`collect_pairwise_data.py`): Her eşleşme için motoru taze başlatıp N round oynatır, her frame'de `(state, action, reward, next_state, terminal)` kaydeder. `ProcessPoolExecutor` ile paralel çalışabilir.
3. **Dataset inşası** (`build_pearl_dataset.py`): 45 dosyayı birleştirip görev-koşullu (task-conditioned) train/val/test split'i oluşturur (ZonerAI/MctsAiZoning = test, PunisherAI/AggressiveTurtle = val, kalan 6 bot = train). Pasif aksiyon (STAND/BACK_STEP/CROUCH/STAND_RECOV) ardışık tekrarlarını downsample eden bir yardımcı da içerir.
4. **Model** (`train_pearl_smoke.py`): `ContextEncoder` (permütasyon-invariant MLP + mean-pool, z_dim=8) ve CQL-tarzı `QNetwork`.
5. **Eğitim** (`train_pearl_full.py`): Checkpoint'li, target-network soft-update'li, gradyan clip'li, erken-patlama tespitli tam eğitim döngüsü. Colab ↔ yerel arasında taşınabilir (göreli yollar).
6. **Değerlendirme**: `evaluate_offline.py` (offline, artan context boyutlarıyla), `baseline_no_context.py` (z=0 baseline), `baseline_opponent_model.py` (context'siz state→action MLP baseline).
7. **Canlı oyun**: `live_ai.py` (headless, toplu tarama: yöntem × rakip), `live_ai_gui.py` (görsel pencereli, tekil round demo).

## Kurulum

```bash
python -m venv ftg-env
ftg-env\Scripts\activate   # Windows
pip install -r requirements.txt
```

DareFightingICE 7.1 motorunu [resmi sayfadan](https://www.ice.ci.ritsumei.ac.jp/~ftgaic/) indirip bu depoyla aynı dizin düzeninde `DareFightingICE-7.1/FightingICE7.1/` altına yerleştirin (motor kendi lisansına tabi olduğu için bu depoya dahil edilmemiştir).

## Kullanım

```bash
# Veri toplama (tüm 45 eşleşme, paralel)
python collect_pairwise_data.py full --rounds 10 --max-workers 3

# Dataset inşası (train/val/test split)
python build_pearl_dataset.py

# Eğitim (checkpoint'ten otomatik devam eder)
python train_pearl_full.py --epochs 300 --cql-alpha 0.3

# Offline değerlendirme
python evaluate_offline.py

# Canlı oyun (headless, tam tarama)
python live_ai.py --rounds 20 --epsilon 0.15
```

## Bulgular (özet)

- **argmax-tuzağı**: Deterministik aksiyon seçimi, Q-değerleri birbirine yakın/düz olduğunda kötü bir aksiyona kilitlenebiliyor. Epsilon-greedy (`--epsilon`) enjeksiyonu bunu kırıp performansı dramatik şekilde artırdı.
- **cql_alpha**: Varsayılan (1.0) yerine daha düşük bir değer (0.3), OOD aksiyonların aşırı bastırılmasını azaltarak genel performansı iyileştirdi.
- **Context-conditioning**: Kolay/scriptli rakiplere (ZonerAI) karşı context'in (`ours` vs `zero`/context-siz) katkısı sınırlıyken, zorlu/adaptif rakiplere (MctsAiZoning, MctsAi23i) karşı tutarlı ve anlamlı bir avantaj sağlıyor — PEARL'ün asıl değerinin ortaya çıktığı yer burası.

## Dizin Yapısı

| Dosya/Klasör | Açıklama |
|---|---|
| `rule_based_bots.py` | Kural-tabanlı rakip botlar |
| `collect_pairwise_data.py` | Paralel veri toplama |
| `build_pearl_dataset.py` | Dataset birleştirme, split, downsampling |
| `train_pearl_smoke.py` | Model tanımları (ContextEncoder, QNetwork, cql_style_loss) |
| `train_pearl_full.py` | Tam eğitim döngüsü |
| `evaluate_offline.py` / `baseline_*.py` | Offline değerlendirme ve baseline'lar |
| `live_ai.py` / `live_ai_gui.py` | Canlı oyun (headless / görsel) |
| `checkpoints/` | Eğitilmiş model checkpoint'leri |
