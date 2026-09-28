"""
pyftg AIInterface kullanan 4 basit kural-tabanli (if-else) bot + karsilastirma
icin bir RandomAI:
  - RusherAI:    mesafeyi hizla kapatip surekli saldirir.
  - TurtleAI:    mesafeyi acik tutar, saldiri gelince blok yapar.
  - ZonerAI:     orta menzili korur, o menzilde uzun menzilli hareket kullanir.
  - PunisherAI:  rakip savunmasizken (control=False / recovery) ceza atar,
                 aksi halde bekler/bloklar.
  - RandomAI:    her karar noktasinda havuzdan rastgele aksiyon secer.

Her biri sadece FrameData'daki mesafe (x koordinati), HP ve rakibin
mevcut Action'ina bakarak karar verir; arama/ogrenme yok.

Tum botlar ortak RuleBasedAI tabanindan gelen log_metrics() ile round
sonunda 4 davranissal metrik raporlar: saldirganlik orani, blok orani,
ortalama mesafe, dash/backstep orani.
"""
import logging
import random

from pyftg import AIInterface, AudioData, CommandCenter, FrameData, GameData, Key, RoundResult, ScreenData
from pyftg.models.enums.action import Action

logger = logging.getLogger(__name__)

# Rakibin "saldiri yapiyor" sayildigi aksiyonlar (Turtle/Punisher icin blok tetikleyici).
ATTACK_ACTIONS = {
    Action.STAND_A, Action.STAND_B, Action.CROUCH_A, Action.CROUCH_B,
    Action.STAND_FA, Action.STAND_FB, Action.CROUCH_FA, Action.CROUCH_FB,
    Action.AIR_A, Action.AIR_B, Action.AIR_DA, Action.AIR_DB,
    Action.AIR_FA, Action.AIR_FB, Action.AIR_UA, Action.AIR_UB,
    Action.THROW_A, Action.THROW_B,
    Action.STAND_D_DF_FA, Action.STAND_D_DF_FB,
    Action.STAND_F_D_DFA, Action.STAND_F_D_DFB,
    Action.STAND_D_DB_BA, Action.STAND_D_DB_BB,
    Action.STAND_D_DF_FC,
}

# Rakibin "savunmasiz/toparlaniyor" sayildigi aksiyonlar (Punisher icin firsat).
RECOVERY_ACTIONS = {
    Action.STAND_RECOV, Action.CROUCH_RECOV, Action.AIR_RECOV,
    Action.STAND_GUARD_RECOV, Action.CROUCH_GUARD_RECOV, Action.AIR_GUARD_RECOV,
    Action.LANDING, Action.DOWN, Action.CHANGE_DOWN, Action.RISE,
    Action.THROW_SUFFER,
}

# Metrik siniflandirmasi icin: bir karar hangi kategoriye giriyor.
GUARD_ACTIONS = {Action.STAND_GUARD, Action.CROUCH_GUARD, Action.AIR_GUARD}
DASH_BACKSTEP_ACTIONS = {Action.DASH, Action.BACK_STEP}

# RandomAI'nin sectigi havuz (test_random_ai.py ile ayni mantik).
RANDOM_ACTION_POOL = [
    Action.FORWARD_WALK, Action.DASH, Action.BACK_STEP, Action.CROUCH,
    Action.JUMP, Action.FOR_JUMP, Action.BACK_JUMP,
    Action.STAND_GUARD, Action.CROUCH_GUARD,
    Action.STAND_A, Action.STAND_B, Action.CROUCH_A, Action.CROUCH_B,
    Action.STAND_FA, Action.STAND_FB, Action.THROW_A, Action.THROW_B,
]


class RuleBasedAI(AIInterface):
    """Ortak pyftg iskeleti; her alt sinif sadece decide() metodunu doldurur."""

    def __init__(self):
        super().__init__()
        self.last_result = None
        self.my_hits = 0
        self.opp_hits = 0
        self._prev_my_hit_confirm = False
        self._prev_opp_hit_confirm = False
        self._distance_sum = 0
        self._distance_count = 0
        self._decision_counts = {"attack": 0, "guard": 0, "dash_backstep": 0, "total": 0}
        self.metrics_history = []  # round_end'de biriken, round basina metrik dict'leri

    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return False

    def initialize(self, game_data: GameData, player_number: bool):
        self.cc = CommandCenter()
        self.key = Key()
        self.player = player_number
        self.frame_data = FrameData()
        self.my_hits = 0
        self.opp_hits = 0
        self._prev_my_hit_confirm = False
        self._prev_opp_hit_confirm = False
        self._distance_sum = 0
        self._distance_count = 0
        self._decision_counts = {"attack": 0, "guard": 0, "dash_backstep": 0, "total": 0}
        logger.info(f"[{self.name()}] initialize: player_number={player_number}, "
                     f"character={game_data.character_names}")

    def get_non_delay_frame_data(self, frame_data: FrameData):
        pass

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        self.cc.set_frame_data(frame_data, self.player)
        self._update_hit_counts(frame_data)
        self._update_distance_stats(frame_data)

    def _update_hit_counts(self, frame_data: FrameData):
        # hit_confirm bir kac frame boyunca True kalabilir; sadece
        # False -> True gecisini (yeni vurus) sayiyoruz.
        if frame_data.empty_flag:
            return
        my = frame_data.get_character(self.player)
        opp = frame_data.get_character(not self.player)
        if my is None or opp is None:
            return

        my_hit_confirm = bool(my.hit_confirm)
        opp_hit_confirm = bool(opp.hit_confirm)
        if my_hit_confirm and not self._prev_my_hit_confirm:
            self.my_hits += 1
        if opp_hit_confirm and not self._prev_opp_hit_confirm:
            self.opp_hits += 1
        self._prev_my_hit_confirm = my_hit_confirm
        self._prev_opp_hit_confirm = opp_hit_confirm

    def _update_distance_stats(self, frame_data: FrameData):
        # Ortalama mesafe metrigi icin: her gecerli frame'de x-mesafesini biriktir.
        if frame_data.empty_flag:
            return
        my = frame_data.get_character(self.player)
        opp = frame_data.get_character(not self.player)
        if my is None or opp is None:
            return
        self._distance_sum += abs(my.x - opp.x)
        self._distance_count += 1

    def get_screen_data(self, screen_data: ScreenData):
        pass

    def get_audio_data(self, audio_data: AudioData):
        pass

    def processing(self):
        if self.frame_data.empty_flag or self.frame_data.current_frame_number <= 0:
            return
        # Onceki komut hala calisiyorsa yeni komut vermeye gerek yok.
        if self.cc.get_skill_flag():
            self.key = self.cc.get_skill_key()
            return

        my = self.frame_data.get_character(self.player)
        opp = self.frame_data.get_character(not self.player)
        if my is None or opp is None:
            return

        distance = abs(my.x - opp.x)
        action = self.decide(my, opp, distance)
        self._record_decision(action)
        self.cc.command_call(action.name)
        self.key = self.cc.get_skill_key()

    def _record_decision(self, action: Action):
        # Sadece yeni verilen komutlari sayiyoruz (bir onceki komut hala
        # calisirken tekrar decide() cagrilmiyor), yani her satir gercek
        # bir "karar noktasi".
        self._decision_counts["total"] += 1
        if action in ATTACK_ACTIONS:
            self._decision_counts["attack"] += 1
        if action in GUARD_ACTIONS:
            self._decision_counts["guard"] += 1
        if action in DASH_BACKSTEP_ACTIONS:
            self._decision_counts["dash_backstep"] += 1

    def decide(self, my, opp, distance: int) -> Action:
        raise NotImplementedError

    def input(self) -> Key:
        return self.key

    def log_metrics(self):
        """Tum botlarin ortak kullandigi 4-metrik raporu: saldirganlik orani,
        blok orani, ortalama mesafe, dash/backstep orani."""
        total = self._decision_counts["total"]
        if total == 0:
            logger.warning(f"[{self.name()}] METRICS: karar verilmedi, metrik hesaplanamadi.")
            return

        aggression_ratio = self._decision_counts["attack"] / total
        block_ratio = self._decision_counts["guard"] / total
        dash_backstep_ratio = self._decision_counts["dash_backstep"] / total
        avg_distance = (self._distance_sum / self._distance_count
                         if self._distance_count > 0 else float("nan"))

        logger.info(
            f"METRICS|{self.name()}|aggression_ratio={aggression_ratio:.3f}"
            f"|block_ratio={block_ratio:.3f}"
            f"|avg_distance={avg_distance:.1f}"
            f"|dash_backstep_ratio={dash_backstep_ratio:.3f}"
            f"|decisions={total}"
        )
        return {
            "aggression_ratio": aggression_ratio,
            "block_ratio": block_ratio,
            "avg_distance": avg_distance,
            "dash_backstep_ratio": dash_backstep_ratio,
            "decisions": total,
        }

    def round_end(self, round_result: RoundResult):
        self.last_result = round_result
        logger.info(f"[{self.name()}] ROUND END: p1_hp={round_result.remaining_hps[0]} "
                     f"p2_hp={round_result.remaining_hps[1]} "
                     f"elapsed_frame={round_result.elapsed_frame} "
                     f"my_hits={self.my_hits} opp_hits={self.opp_hits}")
        m = self.log_metrics()
        if m:
            m["my_hp"] = round_result.remaining_hps[0]
            m["opp_hp"] = round_result.remaining_hps[1]
            self.metrics_history.append(m)

    def game_end(self):
        logger.info(f"[{self.name()}] GAME END")

    def close(self):
        pass


class RusherAI(RuleBasedAI):
    """Mesafeyi hizla kapatir, yakinda surekli saldirir. Dusuk HP'de bile geri cekilmez."""

    CLOSE_RANGE = 60
    MID_RANGE = 150

    def decide(self, my, opp, distance: int) -> Action:
        if distance > self.MID_RANGE:
            return Action.DASH
        if distance > self.CLOSE_RANGE:
            return Action.STAND_FA
        # Yakin mesafe: HP kritikse daha guclu/riskli hareket dene.
        if my.hp < 100:
            return Action.STAND_D_DF_FA
        if opp.action in ATTACK_ACTIONS:
            return Action.THROW_A
        return Action.STAND_B


class TurtleAI(RuleBasedAI):
    """Mesafeyi acik tutar, saldiri gelince bloklar. Onceki surumden daha az
    kacar / daha cok poke atar (BACK_OFF_RANGE dusuruldu, POKE_RANGE genisletildi,
    ondeyken de artik bloga tam gecmiyor)."""

    BACK_OFF_RANGE = 70    # was 100: daha yakina kadar dayaniyor
    POKE_RANGE = 320       # was 250: daha genis aktif poke bolgesi

    def decide(self, my, opp, distance: int) -> Action:
        if opp.action in ATTACK_ACTIONS and distance < self.BACK_OFF_RANGE + 20:
            return Action.CROUCH_GUARD
        if distance < self.BACK_OFF_RANGE:
            return Action.BACK_STEP
        if distance < self.POKE_RANGE:
            # Artik skor farketmeksizin poke atiyor (eskiden onde iken tam blok yapiyordu).
            return Action.CROUCH_B
        return Action.FORWARD_WALK  # was CROUCH: uzakta da hafifce mesafeyi kapatir


class ZonerAI(RuleBasedAI):
    """Orta menzili korur; cok yakinlasirsa acar, cok uzaklasirsa yaklasir,
    o aralikta uzun menzilli saldiri kullanir. Bolge daraltildi ve uzaktan
    yaklasma DASH ile hizlandirildi ki daha az frame "araya girmeden" gecsin."""

    MIN_ZONE = 130   # was 200: daha erken saldiri menziline giriyor
    MAX_ZONE = 280   # was 350: olu bolgeyi daraltiyor

    def decide(self, my, opp, distance: int) -> Action:
        # HP kritikse guvenlik payini biraz artir (eskisi kadar asiri degil).
        min_zone = self.MIN_ZONE + 20 if my.hp < 100 else self.MIN_ZONE
        if distance < min_zone:
            return Action.BACK_STEP
        if distance > self.MAX_ZONE:
            # Cok uzaktaysa DASH ile hizli kapat (was FORWARD_WALK).
            return Action.DASH
        return Action.STAND_D_DF_FA


class PunisherAI(RuleBasedAI):
    """Rakip savunmasiz (control=False ya da recovery aksiyonunda) oldugu an
    ceza saldirisina gecer; onceki surumden farkli olarak firsat yokken de
    artik pasif CROUCH yerine mesafeyi kapatip poke atarak baski kurar."""

    PUNISH_RANGE = 220     # was 150: daha genis ceza penceresi
    POKE_RANGE = 300       # yeni: firsat yokken de bu mesafede poke at

    def decide(self, my, opp, distance: int) -> Action:
        opponent_vulnerable = (not opp.control) or (opp.action in RECOVERY_ACTIONS)
        if opponent_vulnerable:
            if distance < self.PUNISH_RANGE:
                return Action.STAND_D_DF_FA
            return Action.DASH
        if opp.action in ATTACK_ACTIONS and distance < self.PUNISH_RANGE:
            return Action.STAND_GUARD
        # Firsat yok ama pasif kalmak yerine mesafeyi yonet ve poke at.
        if distance > self.POKE_RANGE:
            return Action.FORWARD_WALK
        return Action.CROUCH_B


class RandomAI(RuleBasedAI):
    """Kural-tabanli botlarla karsilastirma icin taban cizgisi: mesafe/HP/aksiyon
    bilgisine hic bakmadan her karar noktasinda havuzdan rastgele aksiyon secer."""

    def decide(self, my, opp, distance: int) -> Action:
        return random.choice(RANDOM_ACTION_POOL)


# ---------------------------------------------------------------------------
# Mevcut 4 botun esik degerlerini degistirerek turetilen varyantlar.
# Amac: ayni "kimlik" (saldirgan/savunmaci/mesafeci/firsatci) etrafinda
# farkli davranissal profiller uretip uretmedigimizi metriklerle gormek.
# ---------------------------------------------------------------------------

class AggressiveTurtle(RuleBasedAI):
    """TurtleAI'nin blok refleksini (rakip saldirirken hala guard) aynen
    korur, ama BACK_OFF_RANGE'i neredeyse sifirlayip POKE_RANGE'i cok
    genisleterek 'geri cekilen' degil 'yerinde durup surekli poke atan'
    bir savunmaci profiline donusturur."""

    BACK_OFF_RANGE = 40     # TurtleAI: 70 -> burada cok daha dar
    POKE_RANGE = 450        # TurtleAI: 320 -> burada cok daha genis

    def decide(self, my, opp, distance: int) -> Action:
        if opp.action in ATTACK_ACTIONS and distance < self.BACK_OFF_RANGE + 20:
            return Action.CROUCH_GUARD
        if distance < self.BACK_OFF_RANGE:
            return Action.BACK_STEP
        if distance < self.POKE_RANGE:
            return Action.STAND_FB if distance > 150 else Action.CROUCH_B
        return Action.FORWARD_WALK


class PassiveRusher(RuleBasedAI):
    """RusherAI'nin saldirganligini azaltir: yakin mesafede bile belirli bir
    olasilikla (RETREAT_PROB) saldirmak yerine geri cekilir -- surekli baski
    yerine arada mesafe acan, daha temkinli bir 'rusher'."""

    CLOSE_RANGE = 60
    MID_RANGE = 150
    RETREAT_PROB = 0.35     # RusherAI'da bu olasilik 0: hic geri cekilmez

    def decide(self, my, opp, distance: int) -> Action:
        if distance > self.MID_RANGE:
            return Action.DASH
        if distance > self.CLOSE_RANGE:
            return Action.STAND_FA
        if random.random() < self.RETREAT_PROB:
            return Action.BACK_STEP
        if opp.action in ATTACK_ACTIONS:
            return Action.THROW_A
        return Action.STAND_B


class CounterPunisher(RuleBasedAI):
    """PunisherAI'nin tetikleyicisi 'rakip savunmasiz (control=False/recovery)'
    iken, bu varyant onun yerine 'rakip ceza menzile girince belirli bir
    olasilikla (COUNTER_PROB)' proaktif saldiriya geciyor -- gercek bir
    firsat penceresi beklemeyen, daha riskli/aceleci bir 'karsi saldiri'."""

    PUNISH_RANGE = 200
    COUNTER_PROB = 0.5

    def decide(self, my, opp, distance: int) -> Action:
        if distance < self.PUNISH_RANGE and random.random() < self.COUNTER_PROB:
            return Action.STAND_D_DF_FA
        if opp.action in ATTACK_ACTIONS and distance < self.PUNISH_RANGE:
            return Action.STAND_GUARD
        if distance > self.PUNISH_RANGE:
            return Action.FORWARD_WALK
        return Action.CROUCH_B


class HybridRusherTurtle(RuleBasedAI):
    """Baslangicta RusherAI gibi agresif oynar; kendi HP'si maksimumun
    %70'inin altina dustugunde TurtleAI'nin savunma mantigina geser.
    (max HP, projede sabit kullanilan --limithp 400 400 varsayimiyla 400.)
    Ilk denemede esik %50 (200 HP) idi ve MctsAi23i'ye karsi tek round'da
    hic tetiklenmemisti (HP 225'te kaldi); %70'e (280 HP) cikarilarak
    savunma moduna gecme ihtimali artirildi."""

    CLOSE_RANGE = 60
    MID_RANGE = 150
    BACK_OFF_RANGE = 70
    POKE_RANGE = 320
    ASSUMED_MAX_HP = 400
    HP_SWITCH_RATIO = 0.7   # was 0.5

    def decide(self, my, opp, distance: int) -> Action:
        defensive = my.hp < self.HP_SWITCH_RATIO * self.ASSUMED_MAX_HP

        if defensive:
            if opp.action in ATTACK_ACTIONS and distance < self.BACK_OFF_RANGE + 20:
                return Action.CROUCH_GUARD
            if distance < self.BACK_OFF_RANGE:
                return Action.BACK_STEP
            if distance < self.POKE_RANGE:
                return Action.CROUCH_B
            return Action.FORWARD_WALK

        # Agresif mod (RusherAI ile ayni mantik).
        if distance > self.MID_RANGE:
            return Action.DASH
        if distance > self.CLOSE_RANGE:
            return Action.STAND_FA
        if opp.action in ATTACK_ACTIONS:
            return Action.THROW_A
        return Action.STAND_B
