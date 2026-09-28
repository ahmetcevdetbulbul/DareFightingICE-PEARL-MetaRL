"""
4 kural-tabanli botu (Rusher, Turtle, Zoner, Punisher) sirayla MctsAi23i'ye
karsi test eder. MctsAi23i, motorun resource-7.1 paketiyle gelen
data/ai/MctsAi23i.jar dosyasindan motor tarafinda otomatik yuklenir
(registered_agents'a eklenmez, sadece isim olarak run_game'e verilir).

On kosul: motor su flag'lerle ayakta olmali:
    --pyftg-mode --headless-mode --limithp 400 400 -r 1

Kullanim:
    ftg-env\\Scripts\\python.exe test_rule_bots.py
"""
import asyncio
import logging

from pyftg.socket.aio.gateway import Gateway

from rule_based_bots import RusherAI, TurtleAI, ZonerAI, PunisherAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPPONENT = "MctsAi23i"
CHARACTER = "ZEN"


async def run_one(bot_cls) -> tuple:
    gateway = Gateway(host="127.0.0.1", port=31415)
    bot = bot_cls()
    bot_name = bot.name()
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
        logger.info(f"--- {bot_name} sonucu: {sonuc} "
                     f"(kendi_hp={my_hp}, {OPPONENT}_hp={opp_hp}) ---\n")
        return (bot_name, sonuc, my_hp, opp_hp)
    logger.warning(f"--- {bot_name}: round sonucu alinamadi ---\n")
    return (bot_name, "BILINMIYOR", None, None)


async def main():
    results = []
    for bot_cls in (RusherAI, TurtleAI, ZonerAI, PunisherAI):
        results.append(await run_one(bot_cls))

    logger.info("=== OZET ===")
    for bot_name, sonuc, my_hp, opp_hp in results:
        logger.info(f"{bot_name:<12} -> {sonuc:<10} "
                     f"(kendi_hp={my_hp}, {OPPONENT}_hp={opp_hp})")


if __name__ == "__main__":
    asyncio.run(main())
