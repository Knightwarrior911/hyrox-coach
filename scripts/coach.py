"""
HYROX Coach Engine
Analyzes real-time HR/HRV data and generates coaching cues via TTS
"""

import asyncio
import json
import time
import math
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from enum import Enum

import edge_tts
import numpy as np


class TrainingPhase(Enum):
    WARMUP = "warmup"
    EASY_RUN = "easy_run"
    TEMPO = "tempo"
    THRESHOLD = "threshold"
    INTERVAL = "interval"
    RECOVERY = "recovery"
    COOLDOWN = "cooldown"
    REST = "rest"


@dataclass
class AthleteProfile:
    max_hr: int = 190
    resting_hr: int = 60
    hyrox_target_pace_min_km: float = 5.0
    hyrox_run_distance_km: float = 1.0
    hyrox_total_runs: int = 8

    @property
    def hrr(self) -> int:
        return self.max_hr - self.resting_hr

    def zone_bounds(self, zone: str) -> tuple:
        zones = {"Z1": (0.50, 0.60), "Z2": (0.60, 0.70), "Z3": (0.70, 0.80), "Z4": (0.80, 0.90), "Z5": (0.90, 1.00)}
        low_pct, high_pct = zones.get(zone, (0.5, 0.6))
        low = int(self.resting_hr + self.hrr * low_pct)
        high = int(self.resting_hr + self.hrr * high_pct)
        return (low, high)


@dataclass
class SessionState:
    phase: TrainingPhase = TrainingPhase.WARMUP
    elapsed_seconds: float = 0
    current_hr: int = 0
    avg_hr: float = 0
    max_hr_session: int = 0
    min_hr_session: int = 999
    rmssd: float = 0
    hr_trend: str = "stable"
    zone: str = "Z1"
    zone_time: dict = field(default_factory=lambda: {f"Z{i}": 0 for i in range(1, 6)})
    total_samples: int = 0
    simulated_run_distance_km: float = 0
    estimated_pace_min_km: float = 0
    last_coach_cue_time: float = 0
    last_zone_alert_time: float = 0
    cues_this_minute: int = 0
    last_cue_minute: int = 0


class HYROXCoach:
    def __init__(self, profile: Optional[AthleteProfile] = None):
        self.profile = profile or AthleteProfile()
        self.state = SessionState()
        self.hr_history: deque = deque(maxlen=300)
        self.rr_history: deque = deque(maxlen=60)
        self.coach_enabled = True
        self.coach_interval_seconds = 30
        self.tts_voice = "en-US-GuyNeural"

    def update(self, hr: int, rr_intervals: list = None, elapsed: float = 0):
        self.state.current_hr = hr
        self.state.elapsed_seconds = elapsed
        self.state.total_samples += 1
        self.hr_history.append(hr)
        if rr_intervals: self.rr_history.extend(rr_intervals)

        hr_arr = np.array(list(self.hr_history))
        self.state.avg_hr = float(np.mean(hr_arr))
        self.state.max_hr_session = max(self.state.max_hr_session, hr)
        self.state.min_hr_session = min(self.state.min_hr_session, hr)
        self.state.zone = self._get_zone(hr)

        if len(hr_arr) >= 60:
            recent = np.mean(hr_arr[-30:])
            previous = np.mean(hr_arr[-60:-30])
            diff = recent - previous
            if diff > 3: self.state.hr_trend = "rising"
            elif diff < -3: self.state.hr_trend = "falling"
            else: self.state.hr_trend = "stable"

        if len(self.rr_history) >= 5:
            rr = np.array(list(self.rr_history))
            diffs = np.diff(rr)
            self.state.rmssd = float(np.sqrt(np.mean(diffs ** 2)))

        self._detect_phase()

    def _get_zone(self, hr: int) -> str:
        for zone in ["Z5", "Z4", "Z3", "Z2", "Z1"]:
            low, high = self.profile.zone_bounds(zone)
            if hr >= low: return zone
        return "Z1"

    def _detect_phase(self):
        hr = self.state.current_hr
        zone = self.state.zone
        elapsed = self.state.elapsed_seconds
        if elapsed < 300:
            self.state.phase = TrainingPhase.WARMUP
        elif zone in ("Z4", "Z5") and self.state.hr_trend == "rising":
            self.state.phase = TrainingPhase.INTERVAL
        elif zone == "Z3":
            self.state.phase = TrainingPhase.TEMPO
        elif zone == "Z2" and self.state.hr_trend == "stable":
            self.state.phase = TrainingPhase.EASY_RUN
        elif zone == "Z1":
            if self.state.avg_hr > self.profile.resting_hr + 30:
                self.state.phase = TrainingPhase.RECOVERY
            else:
                self.state.phase = TrainingPhase.COOLDOWN

    def should_give_cue(self) -> bool:
        now = time.time()
        if now - self.state.last_coach_cue_time < self.coach_interval_seconds: return False
        current_minute = int(now / 60)
        if current_minute != self.state.last_cue_minute:
            self.state.cues_this_minute = 0
            self.state.last_cue_minute = current_minute
        if self.state.cues_this_minute >= 4: return False
        return True

    def generate_cue(self) -> Optional[str]:
        if not self.should_give_cue(): return None
        now = time.time()
        self.state.last_coach_cue_time = now
        self.state.cues_this_minute += 1
        cues = []
        zone = self.state.zone
        target_zone = self._get_target_zone_for_phase()

        if zone != target_zone:
            if zone > target_zone:
                diff = self._zone_distance(zone, target_zone)
                cues.append(self._zone_down_urgent(zone, target_zone) if diff >= 2 else self._zone_down_slight(zone, target_zone))
            else:
                diff = self._zone_distance(target_zone, zone)
                cues.append(self._zone_up_urgent(zone, target_zone) if diff >= 2 else self._zone_up_slight(zone, target_zone))

        if self.state.rmssd > 0:
            if self.state.rmssd < 20 and self.state.phase in (TrainingPhase.INTERVAL, TrainingPhase.THRESHOLD):
                cues.append("Your HRV is low. Focus on controlled breathing. In through nose, out through mouth.")
            elif self.state.rmssd > 50 and self.state.phase == TrainingPhase.RECOVERY:
                cues.append("Good recovery. HRV is coming back up.")

        if self.state.phase in (TrainingPhase.EASY_RUN, TrainingPhase.TEMPO):
            pc = self._hyrox_pacing_cue()
            if pc: cues.append(pc)

        elapsed_min = self.state.elapsed_seconds / 60
        if elapsed_min > 0 and int(elapsed_min) % 5 == 0:
            cues.append(self._status_update())

        return cues[0] if cues else None

    def _get_target_zone_for_phase(self) -> str:
        return {
            TrainingPhase.WARMUP: "Z1", TrainingPhase.EASY_RUN: "Z2",
            TrainingPhase.TEMPO: "Z3", TrainingPhase.THRESHOLD: "Z4",
            TrainingPhase.INTERVAL: "Z4", TrainingPhase.RECOVERY: "Z1",
            TrainingPhase.COOLDOWN: "Z1", TrainingPhase.REST: "Z1",
        }.get(self.state.phase, "Z2")

    def _zone_distance(self, a: str, b: str) -> int:
        return abs(int(a[1]) - int(b[1]))

    def _zone_down_urgent(self, c, t):
        low, high = self.profile.zone_bounds(t)
        return f"Heart rate too high. You're in {c}, target is {t}. Slow down. Aim for {low} to {high} BPM."

    def _zone_down_slight(self, c, t):
        return f"Easy up slightly. Drop it back to {t} zone."

    def _zone_up_urgent(self, c, t):
        low, high = self.profile.zone_bounds(t)
        return f"Push it. You're in {c}, need to get to {t}. Pick up the pace. Target {low} to {high}."

    def _zone_up_slight(self, c, t):
        return f"A bit more effort. Move up to {t}."

    def _hyrox_pacing_cue(self):
        z = self.state.zone
        if z == "Z2": return "Good. This is your HYROX race pace zone. Comfortable but purposeful."
        elif z == "Z3": return "Above HYROX race pace. Fine for training, but race day stay in zone 2."
        elif z == "Z1": return "Too easy for HYROX pace. Pick it up to zone 2."
        return ""

    def _status_update(self):
        em = int(self.state.elapsed_seconds / 60)
        hr = self.state.current_hr
        z = self.state.zone
        avg = round(self.state.avg_hr)
        return f"{em} minutes in. Heart rate {hr}, zone {z}. Average {avg}."

    async def speak(self, text: str):
        try:
            communicate = edge_tts.Communicate(text, self.tts_voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio": audio_data += chunk["data"]
            import base64
            return base64.b64encode(audio_data).decode("utf-8")
        except Exception as e:
            print(f"TTS error: {e}")
            return None

    def get_coaching_analysis(self) -> dict:
        return {
            "phase": self.state.phase.value,
            "zone": self.state.zone,
            "target_zone": self._get_target_zone_for_phase(),
            "hr_trend": self.state.hr_trend,
            "rmssd": round(self.state.rmssd, 1),
            "recovery_score": self._calculate_recovery_score(),
            "training_load": self._calculate_training_load(),
            "hyrox_readiness": self._hyrox_readiness(),
            "suggestions": self._get_suggestions(),
        }

    def _calculate_recovery_score(self) -> int:
        score = 50
        if self.state.rmssd > 0: score += min(30, int(self.state.rmssd / 2))
        if self.state.hr_trend == "falling": score += 10
        elif self.state.hr_trend == "rising": score -= 10
        return max(0, min(100, score))

    def _calculate_training_load(self) -> float:
        if not self.hr_history: return 0
        avg_hr = self.state.avg_hr
        duration_min = self.state.elapsed_seconds / 60
        hr_ratio = (avg_hr - self.profile.resting_hr) / self.profile.hrr
        return round(duration_min * hr_ratio * 0.64 * math.exp(1.92 * hr_ratio), 1)

    def _hyrox_readiness(self) -> dict:
        z2_time = self.state.zone_time.get("Z2", 0)
        high_time = self.state.zone_time.get("Z4", 0) + self.state.zone_time.get("Z5", 0)
        total = sum(self.state.zone_time.values())
        if total == 0: return {"score": 0, "assessment": "No data yet"}
        z2_pct = z2_time / total
        hi_pct = high_time / total
        if z2_pct > 0.6 and hi_pct < 0.15: return {"score": 85, "assessment": "Excellent aerobic base for HYROX"}
        elif z2_pct > 0.4: return {"score": 70, "assessment": "Good aerobic fitness. Build more Z2 volume."}
        elif hi_pct > 0.3: return {"score": 50, "assessment": "Too much high intensity. HYROX needs aerobic base."}
        return {"score": 60, "assessment": "Building fitness. Focus on consistent Z2 training."}

    def _get_suggestions(self) -> list:
        s = []
        if self.state.rmssd < 25 and self.state.elapsed_seconds > 300:
            s.append("HRV is low — consider reducing intensity or taking a recovery day.")
        z1 = self.state.zone_time.get("Z1", 0)
        total = sum(self.state.zone_time.values())
        if total > 0 and z1 / total > 0.5 and self.state.elapsed_seconds > 600:
            s.append("Lots of time in Z1. For HYROX, aim for more Z2 volume.")
        return s
