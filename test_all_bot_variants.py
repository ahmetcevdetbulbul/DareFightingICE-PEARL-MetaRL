"""
rule_based_bots.py'deki 4 orijinal bot + esik degerleri degistirilerek
turetilen 5 yeni varyanti MctsAi23i'ye karsi (bir kez daha) test eder ve
ortak log_metrics() ciktisini (saldirganlik orani, blok orani, ortalama
mesafe, dash/backstep orani) tek bir tabloda yan yana gosterir.

Motor her bot icin sifirdan baslatilir (ayni oturumda ikinci run_game'in
motoru kilitlemesi sorunundan kacinmak icin -- collect_pairwise_data.py'deki
EngineProcess aynen kullanilir).

Kullanim:
    ftg-env\\Scripts\\python.exe test_all_bot_variants.py
"""
import asyncio
import logging

from pyftg.socket.aio.gateway import Gateway

from collect_pairwise_data import CHARACTER, HOST, PORT, EngineProcess, _wait_port_closed
from rule_based_bots import (AggressiveTurtle, CounterPunisher, HybridRusherTurtle,
                              PassiveRusher, PunisherAI, RusherAI, TurtleAI, ZonerAI)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPPONENT = "MctsAi23i"

# Orijinal 4 bot + esik-degistirilmis 5 varyant.
BOTS = {
    "RusherAI": RusherAI,
    "TurtleAI": TurtleAI,
    "ZonerAI": ZonerAI,
    "PunisherAI": PunisherAI,
    "AggressiveTurtle": AggressiveTurtle,
    "PassiveRusher": PassiveRusher,
    "CounterPunisher": CounterPunisher,
    "HybridRusherTurtle": HybridRusherTurtle,
}


async def play_one(bot_name: str, bot_cls) -> dict:
    gateway = Gateway(host=HOST, port=PORT)
    bot = bot_cls()
    gateway.register_ai(bot_name, bot)
    logger.info(f"=== {bot_name} vs {OPPONENT} baslatiliyor ===")
    await gateway.run_game([CHARACTER, CHARACTER], [bot_name, OPPONENT], 1)
    metrics = bot.log_metrics() or {}
    metrics["bot"] = bot_name
    if bot.last_result:
        metrics["my_hp"] = bot.last_result.remaining_hps[0]
        metrics["opp_hp"] = bot.last_result.remaining_hps[1]
    return metrics


def run_one(bot_name: str, bot_cls) -> dict:
    engine = EngineProcess()
    engine.start()
    try:
        metrics = asyncio.run(play_one(bot_name, bot_cls))
    finally:
        engine.stop()
        _wait_port_closed()
    return metrics


def main():
    all_metrics = []
    for bot_name, bot_cls in BOTS.items():
        metrics = run_one(bot_name, bot_cls)
        all_metrics.append(metrics)
        logger.info(f"--- {bot_name} tamamlandi: {metrics} ---\n")

    print("\n=== TUM BOTLARIN PROFILI (MctsAi23i'ye karsi, 1 round) ===")
    header = (f"{'bot':<20} {'aggression':>11} {'block':>8} {'avg_dist':>9} "
              f"{'dash_bstep':>11} {'my_hp':>6} {'opp_hp':>7}")
    print(header)
    for m in all_metrics:
        print(f"{m.get('bot', '?'):<20} "
              f"{m.get('aggression_ratio', float('nan')):>11.3f} "
              f"{m.get('block_ratio', float('nan')):>8.3f} "
              f"{m.get('avg_distance', float('nan')):>9.1f} "
              f"{m.get('dash_backstep_ratio', float('nan')):>11.3f} "
              f"{m.get('my_hp', '?'):>6} {m.get('opp_hp', '?'):>7}")


if __name__ == "__main__":
    main()
