"""
HybridRusherTurtle'i MctsAi23i'ye karsi 5'er round (tek run_game cagrisinda
game_number=5, ayni motor oturumunda -- 5 round boyunca RuleBasedAI.round_end()
her round sonunda kendi metrics_history listesine bir dict ekliyor) calistirip
ORTALAMA profil metriklerini raporlar.

NOT: MidRangeZoner rule_based_bots.py'den kaldirildi (kullanilmayacagi icin);
bu script artik sadece HybridRusherTurtle icin calisir.

Kullanim:
    ftg-env\\Scripts\\python.exe test_5round_variants.py
"""
import asyncio
import logging
import statistics

from pyftg.socket.aio.gateway import Gateway

from collect_pairwise_data import CHARACTER, HOST, PORT, EngineProcess, _wait_port_closed
from rule_based_bots import HybridRusherTurtle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPPONENT = "MctsAi23i"
N_ROUNDS = 5

BOTS = {
    "HybridRusherTurtle": HybridRusherTurtle,
}


async def play_n_rounds(bot_name: str, bot_cls) -> list:
    gateway = Gateway(host=HOST, port=PORT)
    bot = bot_cls()
    gateway.register_ai(bot_name, bot)
    logger.info(f"=== {bot_name} vs {OPPONENT} ({N_ROUNDS} round) baslatiliyor ===")
    await gateway.run_game([CHARACTER, CHARACTER], [bot_name, OPPONENT], N_ROUNDS)
    return bot.metrics_history


def run_bot(bot_name: str, bot_cls) -> list:
    engine = EngineProcess()
    engine.start()
    try:
        history = asyncio.run(play_n_rounds(bot_name, bot_cls))
    finally:
        engine.stop()
        _wait_port_closed()
    return history


def _mean(history, key):
    vals = [h[key] for h in history if key in h]
    return statistics.fmean(vals) if vals else float("nan")


def main():
    all_results = {}
    for bot_name, bot_cls in BOTS.items():
        history = run_bot(bot_name, bot_cls)
        all_results[bot_name] = history
        logger.info(f"--- {bot_name}: {len(history)} round tamamlandi ---")
        for i, h in enumerate(history):
            logger.info(f"    round {i + 1}: aggression={h['aggression_ratio']:.3f} "
                        f"block={h['block_ratio']:.3f} avg_dist={h['avg_distance']:.1f} "
                        f"dash_bstep={h['dash_backstep_ratio']:.3f} "
                        f"hp={h['my_hp']}-{h['opp_hp']}")

    print("\n=== ORTALAMA PROFIL (5 round) ===")
    header = (f"{'bot':<20} {'n':>3} {'aggression':>11} {'block':>8} {'avg_dist':>9} "
              f"{'dash_bstep':>11} {'win_rate':>9} {'avg_hp_diff':>12}")
    print(header)
    for bot_name, history in all_results.items():
        n = len(history)
        wins = sum(1 for h in history if h["my_hp"] > h["opp_hp"])
        hp_diffs = [h["my_hp"] - h["opp_hp"] for h in history]
        print(f"{bot_name:<20} {n:>3} "
              f"{_mean(history, 'aggression_ratio'):>11.3f} "
              f"{_mean(history, 'block_ratio'):>8.3f} "
              f"{_mean(history, 'avg_distance'):>9.1f} "
              f"{_mean(history, 'dash_backstep_ratio'):>11.3f} "
              f"{(wins / n if n else float('nan')):>9.2f} "
              f"{(statistics.fmean(hp_diffs) if hp_diffs else float('nan')):>12.1f}")


if __name__ == "__main__":
    main()
