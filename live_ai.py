"""
Egitilmis PEARL modelini ("bizim yontem"), z=0 baseline'ini ve opponent-model
MLP baseline'ini, test_random_ai.py'nin kullandigi pyftg altyapisina baglayan
CANLI AI scripti.

Her round'un ILK --warmup-seconds saniyesinde (varsayilan 3sn = 180 frame)
YONTEM NE OLURSA OLSUN rastgele/varsayilan hareket edilir (RANDOM_ACTION_POOL);
bu sirada training verisiyle BIREBIR AYNI semada (state/action/reward/next_state,
17 boyutlu state) transitionlar toplanir. Sadece "ours" yontemi bu context'i
gercekten kullanir (encoder'a verip z cikarir); "zero" z'yi sabit sifir yapar,
"opponent_model" context'i hic kullanmaz (sadece state->action). Warmup
bittikten sonra yontemin kendi politikasina gecilir; z (veya siniflandirici
agirliklari) round boyunca SABIT kalir, sadece round basinda bir kere hesaplanir.

ONEMLI: modelin state/action semasi hep "p1 = biz" varsayimiyla toplandigi
icin, canli AI HER ZAMAN player_1 (agents[0]) olarak kaydedilmelidir. Aksi
takdirde ogrenilen politika anlamsizlasir (initialize()'da uyari basilir).

ZonerAI ve MctsAiZoning test split'i tanimlayan iki opponent'tir (train
sirasinda hic gorulmedi) -- bu yuzden burada rakip olarak kullaniliyorlar.

Kullanim:
    # Tek bir hizli test (tek yontem, tek rakip, 1 round):
    ftg-env\\Scripts\\python.exe live_ai.py --method zero --opponent ZonerAI --rounds 1

    # Tam tarama (3 yontem x 2 rakip, varsayilan 3 round/eslesme):
    ftg-env\\Scripts\\python.exe live_ai.py
"""
import argparse
import asyncio
import logging
import random
import socket
import subprocess
import threading
import time
from pathlib import Path

import torch

from pyftg import AIInterface, AudioData, CommandCenter, FrameData, GameData, Key, RoundResult, ScreenData
from pyftg.models.enums.action import Action
from pyftg.socket.aio.gateway import Gateway

from build_pearl_dataset import DEFAULT_DATA_DIR
from train_pearl_smoke import ACTION_DIM, Z_DIM, ContextEncoder, QNetwork
from baseline_opponent_model import OpponentModel
from rule_based_bots import RANDOM_ACTION_POOL, ZonerAI as RuleZonerAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHARACTER = "ZEN"
HOST = "127.0.0.1"
PORT = 31415

ENGINE_DIR = Path(r"C:\Users\USER\Desktop\Paper3\DareFightingICE-7.1\FightingICE7.1")
ENGINE_CLASSPATH = "FightingICE.jar;./lib/*;./lib/lwjgl/*;./lib/lwjgl/natives/windows/amd64/*;./lib/grpc/*;"
ENGINE_CMD = ["java", "-cp", ENGINE_CLASSPATH, "Main",
              "--pyftg-mode", "--headless-mode", "--limithp", "400", "400", "-r", "1"]
ENGINE_READY_MARKER = "Waiting to launch a game"
ENGINE_READY_TIMEOUT = 30

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

METHODS = ["ours", "zero", "opponent_model"]
OPPONENTS = ["ZonerAI", "MctsAiZoning", "MctsAi23i"]  # MctsAi23i egitimde GORULMUS (train split)


# --------------------------------------------------------------------------
# Motor surec yonetimi (collect_pairwise_data.py ile ayni desen)
# --------------------------------------------------------------------------

class EngineProcess:
    def __init__(self):
        self.proc = None
        self._ready_event = threading.Event()
        self._reader_thread = None

    def _read_stdout(self):
        for line in self.proc.stdout:
            if ENGINE_READY_MARKER in line:
                self._ready_event.set()

    def start(self):
        self._ready_event.clear()
        self.proc = subprocess.Popen(
            ENGINE_CMD, cwd=str(ENGINE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()
        if not self._ready_event.wait(timeout=ENGINE_READY_TIMEOUT):
            raise TimeoutError("Motor 'Waiting to launch a game' mesajini vermedi.")
        time.sleep(1.0)
        logger.info("Motor hazir (pid=%s).", self.proc.pid)

    def stop(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        logger.info("Motor durduruldu.")
        self.proc = None


def _wait_port_closed(host=HOST, port=PORT, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) != 0:
                return
        time.sleep(0.3)


# --------------------------------------------------------------------------
# Model yukleme
# --------------------------------------------------------------------------

def load_pearl_models(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    encoder = ContextEncoder()
    q_net = QNetwork()
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    q_net.load_state_dict(ckpt["q_net_state_dict"])
    encoder.eval()
    q_net.eval()
    return encoder, q_net, ckpt.get("epoch")


def load_opponent_model(save_path: Path, data_dir: Path):
    model = OpponentModel()
    if save_path.exists():
        model.load_state_dict(torch.load(save_path, weights_only=False))
        logger.info(f"Opponent model checkpoint'ten yuklendi: {save_path}")
    else:
        logger.info(f"Opponent model checkpoint yok ({save_path}) -- sifirdan egitiliyor...")
        from baseline_opponent_model import train_opponent_model
        from build_pearl_dataset import build_task_tensors
        combined = torch.load(data_dir / "combined.pt", weights_only=False)
        train_rows = [r for r in combined if r["split"] == "train"]
        train_task_tensors = build_task_tensors(train_rows)
        model = train_opponent_model(train_task_tensors, epochs=20, batch_size=256, lr=1e-3, seed=0)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Canli AI
# --------------------------------------------------------------------------

class LiveAI(AIInterface):
    """method: 'ours' | 'zero' | 'opponent_model'."""

    def __init__(self, method: str, encoder=None, q_net=None, opponent_model=None,
                 warmup_frames=180, context_size=64, epsilon=0.0):
        super().__init__()
        assert method in METHODS
        self.method = method
        self.encoder = encoder
        self.q_net = q_net
        # TANI AMACLI, GECICI parametre: epsilon>0 ise argmax yerine
        # epsilon-greedy (epsilon ihtimalle tamamen rastgele aksiyon,
        # 1-epsilon ihtimalle normal argmax) kullanilir -- amac, greedy
        # politikanin dusuk-Q-degerli ama isabetli olabilecek saldiri
        # aksiyonlarini gozardi edip etmedigini test etmek. Varsayilan 0.0
        # ile davranis TAMAMEN degismez (mevcut argmax korunur).
        self.epsilon = epsilon
        self.opponent_model = opponent_model
        self.warmup_frames = warmup_frames
        self.context_size = context_size
        self.last_result = None

    def name(self) -> str:
        return f"LiveAI-{self.method}"

    def is_blind(self) -> bool:
        return False

    def initialize(self, game_data: GameData, player_number: bool):
        if player_number is not True:
            logger.warning(f"[{self.name()}] player_number={player_number} (p2 olarak kaydedildi) -- "
                            "model daima 'p1=biz' varsayimiyla egitildi, sonuclar gecersiz olabilir!")
        self.cc = CommandCenter()
        self.key = Key()
        self.player = player_number
        self.frame_data = FrameData()
        self.frame_count = 0
        self.z = torch.zeros(1, Z_DIM)
        self.context_buffer = []
        self._pending = None
        self.current_state = None
        self._context_finalized = False
        logger.info(f"[{self.name()}] initialize: player_number={player_number}, "
                    f"character={game_data.character_names}")

    def get_non_delay_frame_data(self, frame_data: FrameData):
        pass

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        self.cc.set_frame_data(frame_data, self.player)
        if frame_data.empty_flag:
            return

        my = frame_data.get_character(True)
        opp = frame_data.get_character(False)
        if my is None or opp is None:
            return

        distance = abs(my.x - opp.x)
        state = [
            float(my.hp), float(opp.hp), float(my.x), float(my.y), float(opp.x), float(opp.y),
            float(my.energy), float(opp.energy),
            float(my.state.to_int()), float(opp.state.to_int()),
            float(my.action.to_int()), float(opp.action.to_int()),
            float(my.speed_x), float(my.speed_y), float(opp.speed_x), float(opp.speed_y),
            float(distance),
        ]
        self.current_state = state

        # Onceki frame'in gecisini tamamla (state_t, action_t, reward, state_{t+1}) --
        # training verisiyle ayni sema. Sadece warmup penceresinde biriktiriyoruz.
        if self._pending is not None and self.frame_count <= self.warmup_frames:
            reward = ((self._pending["opp_hp"] - opp.hp) - (self._pending["my_hp"] - my.hp))
            if self._pending["action_taken"] is not None:
                self.context_buffer.append({
                    "state": self._pending["state"], "action": self._pending["action_taken"],
                    "reward": float(reward), "next_state": state,
                })

        self._pending = {"state": state, "my_hp": my.hp, "opp_hp": opp.hp, "action_taken": None}

    def get_screen_data(self, screen_data: ScreenData):
        pass

    def get_audio_data(self, audio_data: AudioData):
        pass

    def processing(self):
        if self.frame_data.empty_flag or self.frame_data.current_frame_number <= 0:
            return
        self.frame_count += 1

        if self.frame_count > self.warmup_frames and not self._context_finalized:
            self._finalize_context()
            self._context_finalized = True

        if self.cc.get_skill_flag():
            self.key = self.cc.get_skill_key()
            return

        if self.frame_count <= self.warmup_frames or self.current_state is None:
            action = random.choice(RANDOM_ACTION_POOL)
        else:
            action = self._policy_action()

        if self._pending is not None:
            self._pending["action_taken"] = action.to_int()

        self.cc.command_call(action.name)
        self.key = self.cc.get_skill_key()

    def _finalize_context(self):
        if self.method != "ours" or not self.context_buffer:
            self.z = torch.zeros(1, Z_DIM)
            return
        n = len(self.context_buffer)
        take = min(n, self.context_size)
        idxs = random.sample(range(n), take)
        c_state = torch.tensor([self.context_buffer[i]["state"] for i in idxs], dtype=torch.float32).unsqueeze(0)
        c_action = torch.tensor([self.context_buffer[i]["action"] for i in idxs], dtype=torch.long).unsqueeze(0)
        c_reward = torch.tensor([self.context_buffer[i]["reward"] for i in idxs], dtype=torch.float32).unsqueeze(0)
        c_next_state = torch.tensor([self.context_buffer[i]["next_state"] for i in idxs], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            self.z = self.encoder(c_state, c_action, c_reward, c_next_state)
        logger.info(f"[{self.name()}] warmup bitti: {take} transition ile z hesaplandi.")

    @torch.no_grad()
    def _policy_action(self) -> Action:
        if self.epsilon > 0.0 and random.random() < self.epsilon:
            return Action.from_int(random.randrange(ACTION_DIM))
        state_t = torch.tensor([self.current_state], dtype=torch.float32)
        if self.method == "opponent_model":
            logits = self.opponent_model(state_t)
            idx = logits.argmax(dim=-1).item()
        else:
            q_all = self.q_net(state_t, self.z)
            idx = q_all.argmax(dim=-1).item()
        return Action.from_int(idx)

    def input(self) -> Key:
        return self.key

    def round_end(self, round_result: RoundResult):
        self.last_result = round_result
        self.frame_count = 0
        self.z = torch.zeros(1, Z_DIM)
        self.context_buffer = []
        self._pending = None
        self._context_finalized = False
        logger.info(f"[{self.name()}] ROUND END p1_hp={round_result.remaining_hps[0]} "
                    f"p2_hp={round_result.remaining_hps[1]}")

    def game_end(self):
        pass

    def close(self):
        pass


# --------------------------------------------------------------------------
# Tek bir eslesmeyi oynatma + toplu tarama
# --------------------------------------------------------------------------

def run_one_match(method: str, opponent: str, n_rounds: int, encoder, q_net, opponent_model,
                   epsilon: float = 0.0) -> list:
    def ai_factory():
        return LiveAI(method=method, encoder=encoder, q_net=q_net, opponent_model=opponent_model,
                       epsilon=epsilon)

    engine = EngineProcess()
    engine.start()

    results = []

    async def _run():
        gateway = Gateway(host=HOST, port=PORT)
        ai = ai_factory()
        gateway.register_ai(method, ai)
        if opponent == "ZonerAI":
            gateway.register_ai("ZonerAI", RuleZonerAI())

        logger.info(f"=== {method} vs {opponent} ({n_rounds} round) baslatiliyor "
                    f"(biz her zaman player_1) ===")

        round_results = []
        orig_round_end = ai.round_end

        def _capture_round_end(round_result):
            round_results.append(round_result)
            orig_round_end(round_result)

        ai.round_end = _capture_round_end

        await gateway.run_game([CHARACTER, CHARACTER], [method, opponent], n_rounds)
        return round_results

    try:
        results = asyncio.run(_run())
    finally:
        engine.stop()
        _wait_port_closed()

    return results


def summarize(method: str, opponent: str, round_results: list) -> dict:
    n = len(round_results)
    if n == 0:
        return {"method": method, "opponent": opponent, "n_rounds": 0,
                "win_rate": float("nan"), "avg_hp_diff": float("nan")}
    wins = 0
    hp_diffs = []
    for r in round_results:
        my_hp, opp_hp = r.remaining_hps[0], r.remaining_hps[1]
        if my_hp > opp_hp:
            wins += 1
        hp_diffs.append(my_hp - opp_hp)
    return {
        "method": method, "opponent": opponent, "n_rounds": n,
        "win_rate": wins / n, "avg_hp_diff": sum(hp_diffs) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR / "checkpoint_latest.pt")
    parser.add_argument("--opponent-model-path", type=Path,
                         default=DEFAULT_CHECKPOINT_DIR / "opponent_model.pt")
    parser.add_argument("--method", choices=METHODS, default=None,
                         help="belirtilmezse ucu de (tam tarama) calisir")
    parser.add_argument("--opponent", choices=OPPONENTS, default=None,
                         help="belirtilmezse ikisi de (tam tarama) calisir")
    parser.add_argument("--rounds", type=int, default=3, help="eslesme basina round sayisi")
    parser.add_argument("--epsilon", type=float, default=0.0,
                         help="TANI AMACLI: >0 ise argmax yerine epsilon-greedy (bu ihtimalle "
                              "tamamen rastgele aksiyon) kullanilir. Varsayilan 0.0 (davranis degismez).")
    args = parser.parse_args()

    encoder, q_net, ckpt_epoch = load_pearl_models(args.checkpoint)
    logger.info(f"PEARL checkpoint: {args.checkpoint.resolve()} (epoch={ckpt_epoch})")
    opponent_model = load_opponent_model(args.opponent_model_path, args.data_dir)

    methods = [args.method] if args.method else METHODS
    opponents = [args.opponent] if args.opponent else OPPONENTS

    summary_rows = []
    for method in methods:
        for opponent in opponents:
            round_results = run_one_match(method, opponent, args.rounds, encoder, q_net, opponent_model,
                                           epsilon=args.epsilon)
            row = summarize(method, opponent, round_results)
            summary_rows.append(row)
            logger.info(f"--- {method} vs {opponent}: win_rate={row['win_rate']:.2f} "
                        f"avg_hp_diff={row['avg_hp_diff']:.1f} ({row['n_rounds']}/{args.rounds} round) ---")

    print("\n=== CANLI MAC OZETI ===")
    print(f"{'method':<16} {'opponent':<14} {'rounds':>7} {'win_rate':>9} {'avg_hp_diff':>12}")
    for row in summary_rows:
        print(f"{row['method']:<16} {row['opponent']:<14} {row['n_rounds']:>7} "
              f"{row['win_rate']:>9.2f} {row['avg_hp_diff']:>12.1f}")


if __name__ == "__main__":
    main()
