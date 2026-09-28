"""
10 bot (8 pyftg botu + 2 motor-gomulu Java AI'si) arasindaki ikili
kombinasyonlar (C(10,2)=45 cift) icin otomatik veri toplama scripti.

Botlar:
  Python (pyftg):  RusherAI, TurtleAI, ZonerAI, PunisherAI, PassiveRusher,
                    CounterPunisher, AggressiveTurtle, HybridRusherTurtle
  Java (motor ici): MctsAi23i, MctsAiZoning

Java-vs-Java eslesmelerde hicbir Python AIInterface motora baglanmaz, bu
yuzden veri toplama pyftg'nin StreamInterface ("spectator"/izleyici)
mekanizmasiyla yapilir: hicbir oyuncuyu kontrol etmeden her iki tarafin
FrameData'sini okur. Bu, ucu turlu eslesme (python-python, python-java,
java-java) icin de ayni kod yolunu kullanmayi saglar.

Her frame icin kaydedilen satir:
    state        : [p1.hp, p2.hp, p1.x, p1.y, p2.x, p2.y, p1.energy,
                     p2.energy, p1.state, p2.state, p1.action, p2.action,
                     p1.speed_x, p1.speed_y, p2.speed_x, p2.speed_y, distance]
    action_p1/p2 : o frame'de karakterin oynadigi Action'in int karsiligi
    reward_p1    : (opp'un kaybettigi hp) - (kendi kaybettigi hp), sifir toplamli,
                     ARTI o frame'de isabetli vurus gerceklestiyse sabit HIT_BONUS
                     (biz vurduysak +HIT_BONUS, biz yedikse -HIT_BONUS) -- HP farki
                     zaten hasara oranli surekli bir sinyal verirken, bu ek terim
                     "isabet oldu mu" sorusuna sabit buyuklukte, zayif vurgularda bile
                     kaybolmayan net/seyrek bir sinyal ekler.
    next_state   : bir sonraki frame'in state'i (round sonunda state ile ayni)
    terminal     : round'un son frame'inde 1.0, aksi halde 0.0
    style_p1/p2  : o taraftaki botun/AI'nin ismi

Motor, her esleyme (pair) icin sifirdan baslatilip N round sonunda kapatilir
(run_game'e game_number=N verilerek tek oturumda N round oynatilir; ayni
oturumda IKINCI bir run_game cagrisi motoru kilitleyen bir sorun oldugu icin
her pair kendi tek run_game() cagrisini alir).

Paralellik: "full" modu ProcessPoolExecutor ile en fazla MAX_WORKERS (=3)
esleşmeyi AYNI ANDA isler. Her worker surec, yasam suresi boyunca SABIT bir
porta (WORKER_PORTS = [31415, 31416, 31417]) sahip kendi motor orneginini
baslatip kapatir -- boylece 3 motor birbirini engellemeden paralel calisir.
RAM zaten yuksek dolulukta oldugu icin MAX_WORKERS bilerek 3'te sinirlandi
(6 cekirdek kaldirsa da). Her esleşme bitince sistem RAM kullanimi ve o
worker surecinin RSS'i loglanir (bellek sikismasini erken yakalamak icin).

Kullanim:
    ftg-env\\Scripts\\python.exe collect_pairwise_data.py test    # 2 cift, 2 round, sirali
    ftg-env\\Scripts\\python.exe collect_pairwise_data.py full    # tum 45 cift (10 bot), cift basina 10 round, 3 paralel worker
    ftg-env\\Scripts\\python.exe collect_pairwise_data.py full --rounds 10 --max-workers 3
"""
import argparse
import asyncio
import itertools
import logging
import multiprocessing
import socket
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import psutil
import torch

from pyftg.aiinterface.stream_interface import StreamInterface
from pyftg.socket.aio.gateway import Gateway

from rule_based_bots import (AggressiveTurtle, CounterPunisher, HybridRusherTurtle, PassiveRusher,
                              PunisherAI, RusherAI, TurtleAI, ZonerAI)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PYTHON_BOTS = {
    "RusherAI": RusherAI,
    "TurtleAI": TurtleAI,
    "ZonerAI": ZonerAI,
    "PunisherAI": PunisherAI,
    "PassiveRusher": PassiveRusher,
    "CounterPunisher": CounterPunisher,
    "AggressiveTurtle": AggressiveTurtle,
    "HybridRusherTurtle": HybridRusherTurtle,
}
JAVA_BOTS = ["MctsAi23i", "MctsAiZoning"]
ALL_BOTS = list(PYTHON_BOTS.keys()) + JAVA_BOTS  # 10 bot -> C(10,2)=45 cift

CHARACTER = "ZEN"
HOST = "127.0.0.1"
PORT = 31415  # sirali/test modunda kullanilan varsayilan port

MAX_WORKERS = 3
WORKER_PORTS = [31415, 31416, 31417]

HIT_BONUS = 2.0  # isabetli vurus basina sabit odul/ceza (bkz. modul docstring'i)

ENGINE_DIR = Path(r"C:\Users\USER\Desktop\Paper3\DareFightingICE-7.1\FightingICE7.1")
ENGINE_CLASSPATH = "FightingICE.jar;./lib/*;./lib/lwjgl/*;./lib/lwjgl/natives/windows/amd64/*;./lib/grpc/*;"
ENGINE_READY_MARKER = "Waiting to launch a game"
ENGINE_READY_TIMEOUT = 30


def engine_cmd(port: int) -> list:
    return [
        "java", "-cp", ENGINE_CLASSPATH, "Main",
        "--pyftg-mode", "--headless-mode", "--limithp", "400", "400", "-r", "1",
        "--port", str(port),
    ]


OUT_DIR = Path(r"C:\Users\USER\Desktop\Paper3\collected_data")


# --------------------------------------------------------------------------
# Motor surec yonetimi
# --------------------------------------------------------------------------

class EngineProcess:
    def __init__(self, port: int = PORT):
        self.port = port
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
            engine_cmd(self.port), cwd=str(ENGINE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()
        if not self._ready_event.wait(timeout=ENGINE_READY_TIMEOUT):
            raise TimeoutError(f"Motor (port={self.port}) 'Waiting to launch a game' mesajini vermedi.")
        # Port acildiktan/mesaj basildiktan sonra kisa bir guvenlik payi.
        time.sleep(1.0)
        logger.info("Motor hazir (pid=%s, port=%s).", self.proc.pid, self.port)

    def stop(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        logger.info("Motor durduruldu (port=%s).", self.port)
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
# Veri toplayan spectator (StreamInterface)
# --------------------------------------------------------------------------

class DataLoggerStream(StreamInterface):
    def __init__(self, style_p1: str, style_p2: str):
        self.style_p1 = style_p1
        self.style_p2 = style_p2
        self.rows = []
        self._pending = None

    def get_frame_data_flag(self) -> bool:
        return True

    def initialize(self, game_data) -> None:
        # Yeni round/game basladi: bekleyen gecisi temizle.
        self._pending = None

    @staticmethod
    def _make_state(p1, p2, distance):
        return [
            float(p1.hp), float(p2.hp), float(p1.x), float(p1.y), float(p2.x), float(p2.y),
            float(p1.energy), float(p2.energy),
            float(p1.state.to_int()), float(p2.state.to_int()),
            float(p1.action.to_int()), float(p2.action.to_int()),
            float(p1.speed_x), float(p1.speed_y), float(p2.speed_x), float(p2.speed_y),
            float(distance),
        ]

    def get_information(self, frame_data) -> None:
        if frame_data.empty_flag:
            return
        p1 = frame_data.get_character(True)
        p2 = frame_data.get_character(False)
        if p1 is None or p2 is None:
            return

        distance = abs(p1.x - p2.x)
        state = self._make_state(p1, p2, distance)

        if self._pending is not None:
            p1_hp_drop = self._pending["_prev_p1_hp"] - p1.hp
            p2_hp_drop = self._pending["_prev_p2_hp"] - p2.hp
            reward_p1 = p2_hp_drop - p1_hp_drop
            if p2_hp_drop > 0:   # biz isabetli vurus yaptik
                reward_p1 += HIT_BONUS
            if p1_hp_drop > 0:   # rakip bize isabetli vurus yapti
                reward_p1 -= HIT_BONUS
            self._pending["reward_p1"] = float(reward_p1)
            self._pending["next_state"] = state
            self._pending["terminal"] = 0.0
            del self._pending["_prev_p1_hp"]
            del self._pending["_prev_p2_hp"]
            self.rows.append(self._pending)

        self._pending = {
            "frame": frame_data.current_frame_number,
            "state": state,
            "action_p1": p1.action.to_int(),
            "action_p2": p2.action.to_int(),
            "style_p1": self.style_p1,
            "style_p2": self.style_p2,
            "_prev_p1_hp": p1.hp,
            "_prev_p2_hp": p2.hp,
        }

    def get_audio_data(self, audio_data) -> None:
        pass

    def get_screen_data(self, screen_data) -> None:
        pass

    def processing(self) -> None:
        pass

    def round_end(self, round_result) -> None:
        if self._pending is not None:
            self._pending["reward_p1"] = 0.0
            self._pending["next_state"] = self._pending["state"]
            self._pending["terminal"] = 1.0
            self._pending.pop("_prev_p1_hp", None)
            self._pending.pop("_prev_p2_hp", None)
            self.rows.append(self._pending)
            self._pending = None

    def game_end(self) -> None:
        pass

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# Tek bir eslesmeyi oynatma
# --------------------------------------------------------------------------

async def play_pair(style_p1: str, style_p2: str, n_rounds: int, port: int) -> list:
    gateway = Gateway(host=HOST, port=port)

    if style_p1 in PYTHON_BOTS:
        gateway.register_ai(style_p1, PYTHON_BOTS[style_p1]())
    if style_p2 in PYTHON_BOTS:
        gateway.register_ai(style_p2, PYTHON_BOTS[style_p2]())

    stream = DataLoggerStream(style_p1, style_p2)
    gateway.register_stream(stream)

    logger.info(f"=== {style_p1} vs {style_p2} ({n_rounds} round) baslatiliyor ===")
    # Stream (spectator) baglantisi keep_alive=True oldugu icin N round bittikten
    # sonra kendiliginden kapanmiyor; bu yuzden run_game'i bekleyip bittiginde
    # stream task'ini biz acikca iptal ediyoruz (yoksa gather() sonsuza kadar bekler).
    stream_task = asyncio.create_task(gateway.start_stream(keep_alive=True))
    await gateway.run_game([CHARACTER, CHARACTER], [style_p1, style_p2], n_rounds)
    stream_task.cancel()
    try:
        await stream_task
    except asyncio.CancelledError:
        pass

    n_terminal = sum(1 for r in stream.rows if r["terminal"] == 1.0)
    logger.info(f"--- {style_p1} vs {style_p2}: {len(stream.rows)} satir, "
                 f"{n_terminal} round tamamlandi (beklenen {n_rounds}) ---")
    return stream.rows


def _log_memory(context: str):
    vm = psutil.virtual_memory()
    proc_rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
    logger.info(f"[BELLEK] {context} | sistem_RAM=%{vm.percent:.1f} "
                f"(kullanilan={vm.used / 1e9:.1f}GB / toplam={vm.total / 1e9:.1f}GB, "
                f"musait={vm.available / 1e9:.1f}GB) | bu_surec_RSS={proc_rss_mb:.0f}MB")


def _pair_already_done(style_p1: str, style_p2: str, n_rounds: int, out_dir: Path) -> bool:
    """Daha once (ör. yarida kalan bir kosuda) tamamlanmis bir cifti atlamak
    icin: dosya var mi VE beklenen round sayisi kadar terminal=1.0 satiri
    iceriyor mu diye dogrular (sadece dosya varligina degil)."""
    path = out_dir / f"{style_p1}_vs_{style_p2}.pt"
    if not path.exists():
        return False
    try:
        rows = torch.load(path, weights_only=False)
    except Exception:
        return False
    n_terminal = sum(1 for r in rows if r.get("terminal") == 1.0)
    return n_terminal == n_rounds


def run_one_pair(style_p1: str, style_p2: str, n_rounds: int, out_dir: Path, port: int = PORT) -> list:
    engine = EngineProcess(port=port)
    engine.start()
    try:
        rows = asyncio.run(play_pair(style_p1, style_p2, n_rounds, port))
    finally:
        engine.stop()
        _wait_port_closed(port=port)

    if not rows:
        logger.error(f"!!! {style_p1} vs {style_p2}: HIC VERI TOPLANMADI (bos sonuc)")
        _log_memory(f"{style_p1} vs {style_p2} (bos sonuc, port={port})")
        return rows

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{style_p1}_vs_{style_p2}.pt"
    torch.save(rows, out_path)
    logger.info(f"Kaydedildi: {out_path} ({len(rows)} satir)")
    _log_memory(f"{style_p1} vs {style_p2} tamamlandi (port={port})")
    return rows


# --------------------------------------------------------------------------
# Toplu calistirma -- sirali (test modu icin)
# --------------------------------------------------------------------------

def run_batch(pairs, n_rounds: int, out_dir: Path):
    summary = []
    for style_p1, style_p2 in pairs:
        rows = run_one_pair(style_p1, style_p2, n_rounds, out_dir)
        n_terminal = sum(1 for r in rows if r.get("terminal") == 1.0)
        summary.append((style_p1, style_p2, len(rows), n_terminal))

    logger.info("=== TOPLU OZET ===")
    for style_p1, style_p2, n_rows, n_terminal in summary:
        status = "OK" if n_rows > 0 and n_terminal == n_rounds else "SORUNLU"
        logger.info(f"[{status}] {style_p1:<12} vs {style_p2:<12} -> "
                     f"{n_rows:>5} satir, {n_terminal}/{n_rounds} round")
    return summary


# --------------------------------------------------------------------------
# Toplu calistirma -- paralel (full modu icin, ProcessPoolExecutor)
# --------------------------------------------------------------------------

_worker_port = None  # her worker surecinde initializer ile bir kez atanan sabit port


def _init_worker(port_queue):
    global _worker_port
    _worker_port = port_queue.get()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info(f"[WORKER] baslatildi, sabit port={_worker_port} (pid={psutil.Process().pid})")


def _run_pair_worker(args):
    style_p1, style_p2, n_rounds, out_dir_str = args
    out_dir = Path(out_dir_str)
    rows = run_one_pair(style_p1, style_p2, n_rounds, out_dir, port=_worker_port)
    n_terminal = sum(1 for r in rows if r.get("terminal") == 1.0)
    return style_p1, style_p2, len(rows), n_terminal


def run_batch_parallel(pairs, n_rounds: int, out_dir: Path, max_workers: int = MAX_WORKERS,
                        ports: list = None):
    ports = ports or WORKER_PORTS
    if len(ports) < max_workers:
        raise ValueError(f"max_workers={max_workers} icin en az o kadar port gerekli "
                          f"(WORKER_PORTS={ports}).")

    # Daha once (ör. yarida kalan bir kosuda) tamamlanmis ciftleri atla.
    todo = []
    skipped = []
    for style_p1, style_p2 in pairs:
        if _pair_already_done(style_p1, style_p2, n_rounds, out_dir):
            skipped.append((style_p1, style_p2))
        else:
            todo.append((style_p1, style_p2, n_rounds, str(out_dir)))
    if skipped:
        logger.info(f"{len(skipped)} cift zaten tamamlanmis, atlaniyor: "
                     f"{[f'{a} vs {b}' for a, b in skipped]}")
    logger.info(f"{len(todo)} cift islenecek, max_workers={max_workers}, portlar={ports[:max_workers]}.")

    manager = multiprocessing.Manager()
    port_queue = manager.Queue()
    for p in ports[:max_workers]:
        port_queue.put(p)

    summary = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker,
                              initargs=(port_queue,)) as executor:
        futures = {executor.submit(_run_pair_worker, task): task for task in todo}
        for future in as_completed(futures):
            style_p1, style_p2, n_rounds_task, _ = futures[future]
            try:
                result = future.result()
                summary.append(result)
                logger.info(f"[BITTI] {result[0]} vs {result[1]}: {result[2]} satir, "
                             f"{result[3]}/{n_rounds_task} round "
                             f"({len(summary)}/{len(todo)} cift islendi)")
            except Exception as exc:
                logger.error(f"!!! {style_p1} vs {style_p2} worker'da hata: {exc}")
                summary.append((style_p1, style_p2, 0, 0))

    logger.info("=== TOPLU OZET (paralel) ===")
    for style_p1, style_p2, n_rows, n_terminal in summary:
        status = "OK" if n_rows > 0 and n_terminal == n_rounds else "SORUNLU"
        logger.info(f"[{status}] {style_p1:<18} vs {style_p2:<18} -> "
                     f"{n_rows:>6} satir, {n_terminal}/{n_rounds} round")
    return summary


def recompute_rewards(out_dir: Path, hit_bonus: float = HIT_BONUS) -> None:
    """Motoru YENIDEN CALISTIRMADAN, out_dir'deki her *.pt dosyasinin
    reward_p1'ini gunceller: state[0]/state[1] = p1.hp/p2.hp ve
    next_state[0]/next_state[1] ayni frame cifti icin zaten kayitli
    oldugundan, HP dususu (dolayisiyla isabet olup olmadigi) bu alanlardan
    yeniden turetilebilir -- reward formulunun degismesi (ör. HIT_BONUS
    eklenmesi) 45 cift x 10 round'luk ~3 saatlik veri toplamayi tekrar
    gerektirmez, sadece bu fonksiyonu calistirmak yeterlidir.

    NOT: terminal=1.0 satirlarda next_state==state (round_end'de boyle
    kaydedildigi icin) oldugundan HP farki 0 cikar, bu satirlar dogal
    olarak etkilenmez (reward 0.0 kalir) -- ayrica bir ozel durum gerekmez.
    """
    files = sorted(f for f in Path(out_dir).glob("*.pt") if f.name != "combined.pt")
    logger.info(f"{len(files)} dosyada reward_p1 yeniden hesaplanacak (HIT_BONUS={hit_bonus}).")
    for f in files:
        rows = torch.load(f, weights_only=False)
        n_changed = 0
        for r in rows:
            p1_hp_drop = r["state"][0] - r["next_state"][0]
            p2_hp_drop = r["state"][1] - r["next_state"][1]
            new_reward = p2_hp_drop - p1_hp_drop
            if p2_hp_drop > 0:
                new_reward += hit_bonus
            if p1_hp_drop > 0:
                new_reward -= hit_bonus
            new_reward = float(new_reward)
            if new_reward != r["reward_p1"]:
                n_changed += 1
            r["reward_p1"] = new_reward
        torch.save(rows, f)
        logger.info(f"  {f.name}: {len(rows)} satir, {n_changed} tanesinin reward'i degisti")
    logger.info("Bitti. Simdi build_pearl_dataset.py'yi calistirip combined.pt'i yeniden olusturun.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["test", "full", "recompute-rewards"])
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                         help=f"full modunda kac esleşme paralel islensin (varsayilan {MAX_WORKERS})")
    args = parser.parse_args()

    if args.mode == "recompute-rewards":
        recompute_rewards(OUT_DIR)
    elif args.mode == "test":
        # En riskli iki farkli kod yolunu kapsayacak sekilde secildi:
        #   1) python vs java (daha once dogrulanmis AIController yolu)
        #   2) java vs java (SADECE spectator/stream yoluyla veri toplanan,
        #      hic AIController baglanmayan, daha once hic test edilmemis yol)
        pairs = [
            ("RusherAI", "MctsAi23i"),
            ("MctsAi23i", "MctsAiZoning"),
        ]
        n_rounds = args.rounds or 2
        logger.info(f"Mod={args.mode}, {len(pairs)} cift, cift basina {n_rounds} round (sirali).")
        run_batch(pairs, n_rounds, OUT_DIR)
    else:
        pairs = list(itertools.combinations(ALL_BOTS, 2))
        n_rounds = args.rounds or 10
        logger.info(f"Mod={args.mode}, {len(pairs)} cift, cift basina {n_rounds} round, "
                     f"max_workers={args.max_workers} (paralel).")
        run_batch_parallel(pairs, n_rounds, OUT_DIR, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
