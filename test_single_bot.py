"""
Tek bir kural-tabanli botu MctsAi23i'ye karsi bir round test eder.
Ardisik maclarda motorun MctsAi23i'yi ikinci kez yuklerken kilitlenmesi
sorunundan kacinmak icin her bot icin motor ayri baslatilir.

Kullanim:
    ftg-env\\Scripts\\python.exe test_single_bot.py <BotAdi>
    (gecerli isimler icin BOTS sozlugune bakin)
"""
import asyncio
import logging
import sys

from pyftg.socket.aio.gateway import Gateway

from rule_based_bots import (AggressiveTurtle, CounterPunisher, HybridRusherTurtle,
                              PassiveRusher, PunisherAI, RandomAI, RusherAI, TurtleAI, ZonerAI)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOTS = {
    "RusherAI": RusherAI,
    "TurtleAI": TurtleAI,
    "ZonerAI": ZonerAI,
    "PunisherAI": PunisherAI,
    "RandomAI": RandomAI,
    "AggressiveTurtle": AggressiveTurtle,
    "PassiveRusher": PassiveRusher,
    "CounterPunisher": CounterPunisher,
    "HybridRusherTurtle": HybridRusherTurtle,
}
OPPONENT = "MctsAi23i"
CHARACTER = "ZEN"


async def main(bot_name: str):
    bot_cls = BOTS[bot_name]
    gateway = Gateway(host="127.0.0.1", port=31415)
    bot = bot_cls()
    gateway.register_ai(bot_name, bot)

    logger.info(f"=== {bot_name} vs {OPPONENT} baslatiliyor ===")
    await gateway.run_game(
        characters=[CHARACTER, CHARACTER],
        agents=[bot_name, OPPONENT],
        game_number=1,
    )

    if bot.last_result:
        my_hp = bot.last_result.remaining_hps[0]
        opp_hp = bot.last_result.remaining_hps[1]
        sonuc = "KAZANDI" if my_hp > opp_hp else ("KAYBETTI" if my_hp < opp_hp else "BERABERE")
        logger.info(f"SONUC|{bot_name}|{sonuc}|{my_hp}|{opp_hp}|"
                     f"my_hits={bot.my_hits}|opp_hits={bot.opp_hits}")
    else:
        logger.warning(f"SONUC|{bot_name}|BILINMIYOR|None|None|my_hits={bot.my_hits}|opp_hits={bot.opp_hits}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
