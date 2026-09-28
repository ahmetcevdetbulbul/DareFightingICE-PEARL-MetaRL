"""
live_ai.py'nin GUI (headless OLMAYAN) versiyonu: motoru --headless-mode
OLMADAN baslatir, TEK round oynatir -- "ours" (PEARL) P1 tarafinda,
secilen rakip P2 tarafinda. Amac ekranda P1/P2'yi gorsel olarak ayirt
etmek (motor kurallarinca P1 sol/varsayilan, P2 sag/alternatif paletle
cizilir).

live_ai.py'deki LiveAI sinifini ve model yukleme fonksiyonlarini aynen
kullanir; sadece motor komutundan --headless-mode cikarilir ve tek
esleşme / tek round icin sadelestirilmistir.

Rakip secimi: ZonerAI ve RusherAI Python tarafinda (rule_based_bots.py)
tanimli oldugu icin gateway'e kayit edilir; MctsAiZoning/MctsAi23i motorun
kendi yerlesik Java botlari oldugu icin sadece isimle run_game'e verilir.

Kullanim:
    ftg-env\\Scripts\\python.exe live_ai_gui.py --checkpoint checkpoints/checkpoint_epoch_0099.pt --opponent ZonerAI
"""
import argparse
import asyncio
import logging
import socket
import subprocess
import threading
import time
from pathlib import Path

from pyftg.socket.aio.gateway import Gateway

from live_ai import (CHARACTER, DEFAULT_CHECKPOINT_DIR, ENGINE_DIR, ENGINE_READY_TIMEOUT, HOST,
                      PORT, LiveAI, load_pearl_models)
from rule_based_bots import RusherAI as RuleRusherAI
from rule_based_bots import ZonerAI as RuleZonerAI

# Python tarafinda tanimli (gateway'e kayit gereken) rakipler; digerleri
# (ornegin MctsAiZoning, MctsAi23i) motorun yerlesik Java botlaridir --
# sadece run_game'e isimle verilir, register_ai gerekmez.
PYTHON_OPPONENTS = {"ZonerAI": RuleZonerAI, "RusherAI": RuleRusherAI}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --headless-mode YOK -- motor gorsel pencere acar.
ENGINE_CLASSPATH = "FightingICE.jar;./lib/*;./lib/lwjgl/*;./lib/lwjgl/natives/windows/amd64/*;./lib/grpc/*;"
GUI_ENGINE_CMD = ["java", "-cp", ENGINE_CLASSPATH, "Main",
                   "--pyftg-mode", "--limithp", "400", "400", "-r", "1"]
ENGINE_READY_MARKER = "Waiting to launch a game"


class GuiEngineProcess:
    def __init__(self):
        self.proc = None
        self._ready_event = threading.Event()
        self._reader_thread = None

    def _read_stdout(self):
        for line in self.proc.stdout:
            print("[ENGINE]", line.rstrip())
            if ENGINE_READY_MARKER in line:
                self._ready_event.set()

    def start(self):
        self._ready_event.clear()
        self.proc = subprocess.Popen(
            GUI_ENGINE_CMD, cwd=str(ENGINE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()
        if not self._ready_event.wait(timeout=ENGINE_READY_TIMEOUT):
            raise TimeoutError("Motor 'Waiting to launch a game' mesajini vermedi (GUI modu).")
        time.sleep(1.5)
        logger.info("Motor (GUI) hazir (pid=%s).", self.proc.pid)

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


async def _run_single_round(encoder, q_net, opponent: str):
    gateway = Gateway(host=HOST, port=PORT)
    ai = LiveAI(method="ours", encoder=encoder, q_net=q_net)
    gateway.register_ai("ours", ai)
    if opponent in PYTHON_OPPONENTS:
        gateway.register_ai(opponent, PYTHON_OPPONENTS[opponent]())

    logger.info(f"=== ours (P1) vs {opponent} (P2), 1 round, GUI modu ===")
    await gateway.run_game([CHARACTER, CHARACTER], ["ours", opponent], 1)
    return ai.last_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR / "checkpoint_latest.pt")
    parser.add_argument("--opponent", default="ZonerAI")
    args = parser.parse_args()

    encoder, q_net, ckpt_epoch = load_pearl_models(args.checkpoint)
    logger.info(f"PEARL checkpoint: {args.checkpoint.resolve()} (epoch={ckpt_epoch})")

    engine = GuiEngineProcess()
    engine.start()
    try:
        result = asyncio.run(_run_single_round(encoder, q_net, args.opponent))
    finally:
        engine.stop()
        _wait_port_closed()

    if result:
        logger.info(f"SONUC: p1(ours)_hp={result.remaining_hps[0]} p2({args.opponent})_hp={result.remaining_hps[1]}")


if __name__ == "__main__":
    main()
