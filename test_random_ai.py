"""
DareFightingICE 7.1 motoruna pyftg (gRPC) ile bağlanan, rastgele aksiyon
seçen minimal test AI'ı. Tek round oynar (motor -r 1 ile başlatıldığı için)
ve bağlantı/round akışının çalıştığını doğrular.

Kullanım:
    ftg-env\\Scripts\\python.exe test_random_ai.py
"""
import asyncio
import json
import logging
import random

from pyftg import AIInterface, CommandCenter, GameData, FrameData, Key, RoundResult
from pyftg.models.enums.action import Action
from pyftg.socket.aio.gateway import Gateway

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Karakterin sahip olduğu discrete aksiyon havuzu (CommandCenter.action_to_command
# içindeki isimlerle birebir eşleşmeli).
ACTIONS = [
    Action.FORWARD_WALK, Action.DASH, Action.BACK_STEP, Action.CROUCH,
    Action.JUMP, Action.FOR_JUMP, Action.BACK_JUMP,
    Action.STAND_GUARD, Action.CROUCH_GUARD,
    Action.STAND_A, Action.STAND_B, Action.CROUCH_A, Action.CROUCH_B,
    Action.STAND_FA, Action.STAND_FB, Action.THROW_A, Action.THROW_B,
]


class RandomAI(AIInterface):
    def __init__(self):
        self.command_center = CommandCenter()
        self.key = Key()
        self.frame_data = FrameData()
        self.player_number = False
        self.round_count = 0
        self._state_debug_printed = False

    def name(self) -> str:
        return "RandomAI"

    def is_blind(self) -> bool:
        return False

    def initialize(self, game_data: GameData, player_number: bool):
        self.game_data = game_data
        self.player_number = player_number
        self.command_center = CommandCenter()
        self.key = Key()
        logger.info(f"initialize: player_number={player_number}, "
                     f"character={game_data.character_names}")

    def get_non_delay_frame_data(self, frame_data: FrameData):
        pass

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        self.command_center.set_frame_data(frame_data, self.player_number)

        # Debug: ilk dolu frame'de state objesinin tüm alanlarını ve
        # mevcut aksiyon enum'unu bir kere yazdır.
        if not self._state_debug_printed and not frame_data.empty_flag:
            self._state_debug_printed = True
            logger.info(
                f"[DEBUG p{int(self.player_number) + 1}] "
                f"FrameData(frame={frame_data.current_frame_number}, "
                f"round={frame_data.current_round}) tüm alanlar:\n"
                + json.dumps(frame_data.to_dict(), indent=2, ensure_ascii=False)
            )
            p1_char = frame_data.get_character(True)
            p2_char = frame_data.get_character(False)
            logger.info(
                f"[DEBUG p{int(self.player_number) + 1}] mevcut aksiyon enum'u -> "
                f"p1={p1_char.action.name if p1_char else None} "
                f"(int={p1_char.action.to_int() if p1_char else None}), "
                f"p2={p2_char.action.name if p2_char else None} "
                f"(int={p2_char.action.to_int() if p2_char else None})"
            )
            logger.info(
                "[DEBUG] Action enum'unun tüm olası değerleri: "
                + ", ".join(f"{a.name}={a.to_int()}" for a in Action)
            )

    def get_screen_data(self, screen_data):
        pass

    def get_audio_data(self, audio_data):
        pass

    def processing(self):
        # Round bittiyse veya henüz veri gelmediyse yeni komut vermeye gerek yok.
        if self.frame_data.empty_flag:
            return
        if not self.command_center.get_skill_flag():
            action = random.choice(ACTIONS)
            logger.info(f"frame={self.frame_data.current_frame_number} "
                         f"p{int(self.player_number)+1} action -> {action.name}")
            self.command_center.command_call(action.name)
        self.key = self.command_center.get_skill_key()

    def input(self) -> Key:
        return self.key

    def round_end(self, round_result: RoundResult):
        self.round_count += 1
        logger.info(f"ROUND END #{self.round_count} "
                     f"(round={round_result.current_round}): "
                     f"p1_hp={round_result.remaining_hps[0]} "
                     f"p2_hp={round_result.remaining_hps[1]} "
                     f"elapsed_frame={round_result.elapsed_frame}")

    def game_end(self):
        logger.info("GAME END")

    def close(self):
        pass


async def main():
    gateway = Gateway(host="127.0.0.1", port=31415)
    gateway.register_ai("RandomAI", RandomAI())
    gateway.register_ai("RandomAI2", RandomAI())

    logger.info("Motora bağlanılıyor ve tek round başlatılıyor...")
    await gateway.run_game(
        characters=["ZEN", "ZEN"],
        agents=["RandomAI", "RandomAI2"],
        game_number=1,
    )
    logger.info("Bağlantı doğrulandı, oyun tamamlandı.")


if __name__ == "__main__":
    asyncio.run(main())
