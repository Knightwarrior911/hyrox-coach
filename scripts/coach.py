"""
coach.py - Multi-sport live coaching engine.

Turns a stream of heart-rate / RR samples into:
  * meaningful real-time analysis (zones, HRV, cardiac drift, HR recovery,
    training load, sport-specific readiness), and
  * spoken coaching cues (TTS) that are specific, prioritised and varied.

Sports (set per session via set_sport):
  HYROX    - steady compromised-running pacing + aerobic discipline
  FOOTBALL - intermittent high intensity: sprint / recovery, repeat-sprint ability
  GENERIC  - zone-based coaching for any cardio session

server.py calls:
  coach.set_sport("football")     # optional, on session start
  coach.reset()                   # clear state for a new session
  coach.update(hr, rr_intervals)  # every sample
  coach.generate_cue()            # -> text or None (call on the cue loop)
  coach.get_coaching_analysis()   # -> dict for the dashboard
  await coach.speak(text)         # -> base64 mp3 for the browser
"""

import time
import math
import random
import base64
from dataclasses import dataclass
from collections import deque
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import edge_tts


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Sport(Enum):
    HYROX = "hyrox"
    FOOTBALL = "football"
    GENERIC = "generic"


class Phase(Enum):
    WARMUP = "warmup"
    STEADY = "steady"        # aerobic / compromised running
    TEMPO = "tempo"
    THRESHOLD = "threshold"
    SPRINT = "sprint"        # football high-intensity burst
    RECOVERY = "recovery"    # active recovery between efforts
    COOLDOWN = "cooldown"


@dataclass
class AthleteProfile:
    max_hr: int = 190
    resting_hr: int = 60
    hyrox_target_pace_min_km: float = 5.0

    @property
    def hrr(self) -> int:
        return max(1, self.max_hr - self.resting_hr)

    def zone_bounds(self, zone: str) -> tuple:
        bands = {"Z1": (0.50, 0.60), "Z2": (0.60, 0.70),
                 "Z3": (0.70, 0.80), "Z4": (0.80, 0.90), "Z5": (0.90, 1.00)}
        lo, hi = bands.get(zone, (0.5, 0.6))
        return (int(self.resting_hr + self.hrr * lo),
                int(self.resting_hr + self.hrr * hi))

    def zone_for(self, hr: int) -> str:
        pct = (hr - self.resting_hr) / self.hrr
        if pct >= 0.90: return "Z5"
        if pct >= 0.80: return "Z4"
        if pct >= 0.70: return "Z3"
        if pct >= 0.60: return "Z2"
        return "Z1"


@dataclass
class Cue:
    priority: int      # 0 = highest (safety)
    key: str           # cooldown bucket
    interval: float    # min seconds between cues of this key
    text: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class LiveCoach:
    ZONE_LABELS = {"Z1": "recovery", "Z2": "aerobic", "Z3": "tempo",
                   "Z4": "threshold", "Z5": "max"}

    VOICES = {Sport.HYROX: "en-US-GuyNeural",
              Sport.FOOTBALL: "en-US-ChristopherNeural",
              Sport.GENERIC: "en-US-GuyNeural"}

    def __init__(self, profile: Optional[AthleteProfile] = None,
                 sport: Sport = Sport.HYROX):
        self.profile = profile or AthleteProfile()
        self.sport = sport
        self.coach_enabled = True
        self.min_cue_spacing = 8.0
        self.coach_interval_seconds = 30
        self.tts_voice = self.VOICES[sport]
        self.reset()

    # -- lifecycle ------------------------------------------------------

    def reset(self):
        now = time.time()
        self.session_start = now
        self._last_ts = now
        self.hr_history: deque = deque(maxlen=3600)
        self.rr_history: deque = deque(maxlen=300)
        self.zone_time: Dict[str, float] = {f"Z{i}": 0.0 for i in range(1, 6)}
        self.current_hr = 0
        self.avg_hr = 0.0
        self.max_hr_session = 0
        self.min_hr_session = 999
        self.zone = "Z1"
        self.phase = Phase.WARMUP
        self.hr_trend = "stable"
        self.rmssd = 0.0
        self.sdnn = 0.0
        self.pnn50 = 0.0
        self.lf_hf = 0.0
        self.drift_bpm_per_min = 0.0
        self.hr_recovery_60s = 0.0
        self.training_load = 0.0
        self.sprint_count = 0
        self._in_sprint = False
        self._sprint_peak_hr = 0
        self._sprint_start_ts = 0.0
        self._sprint_end_ts = 0.0
        self._sprint_end_peak = 0
        self._pending_recovery = False
        self.last_sprint_recovery = 0.0
        self.work_time = 0.0
        self.rest_time = 0.0
        self.recovery_trend = "stable"
        self._recovery_samples: deque = deque(maxlen=6)
        self._cue_last: Dict[str, float] = {}
        self._last_cue_ts = 0.0
        self._last_cue_text = ""
        self.last_insight = ""

    def set_sport(self, sport):
        if isinstance(sport, str):
            try:
                sport = Sport(sport.lower())
            except ValueError:
                sport = Sport.GENERIC
        self.sport = sport
        self.tts_voice = self.VOICES.get(sport, "en-US-GuyNeural")

    # -- ingest ---------------------------------------------------------

    def update(self, hr: int, rr_intervals: Optional[List[int]] = None):
        if not hr or hr <= 0:
            return
        now = time.time()
        dt = now - self._last_ts
        self._last_ts = now

        self.current_hr = int(hr)
        self.hr_history.append((now, self.current_hr))
        if rr_intervals:
            self.rr_history.extend([r for r in rr_intervals if 250 < r < 2000])

        hrs = np.array([h for _, h in self.hr_history], dtype=float)
        self.avg_hr = float(hrs.mean())
        self.max_hr_session = max(self.max_hr_session, self.current_hr)
        self.min_hr_session = min(self.min_hr_session, self.current_hr)
        self.zone = self.profile.zone_for(self.current_hr)

        if 0 < dt < 30:
            self.zone_time[self.zone] += dt
            if self.zone in ("Z4", "Z5"):
                self.work_time += dt
            elif self.zone in ("Z1", "Z2"):
                self.rest_time += dt
            self._accumulate_load(dt)

        self._update_trend(hrs)
        self._update_hrv()
        self._update_drift(now)
        self._update_recovery(now)
        self._detect_phase()
        self._track_sprints(now)

    # -- analytics ------------------------------------------------------

    def _update_trend(self, hrs):
        if len(hrs) >= 60:
            d = hrs[-20:].mean() - hrs[-60:-20].mean()
            self.hr_trend = "rising" if d > 3 else "falling" if d < -3 else "stable"
        else:
            self.hr_trend = "stable"

    def _update_hrv(self):
        if len(self.rr_history) >= 10:
            rr = np.array(self.rr_history, dtype=float)
            diffs = np.diff(rr)
            self.rmssd = float(np.sqrt(np.mean(diffs ** 2)))
            self.sdnn = float(np.std(rr))
            self.pnn50 = float(np.mean(np.abs(diffs) > 50) * 100)
            self.lf_hf = self._lf_hf(rr)

    def _lf_hf(self, rr):
        if len(rr) < 64:
            return 0.0
        x = rr - rr.mean()
        fs = 1000.0 / rr.mean()
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
        psd = np.abs(np.fft.rfft(x)) ** 2
        lf = psd[(freqs >= 0.04) & (freqs < 0.15)].sum()
        hf = psd[(freqs >= 0.15) & (freqs < 0.40)].sum()
        return float(lf / hf) if hf > 0 else 0.0

    def _update_drift(self, now):
        window = [(t, h) for t, h in self.hr_history if now - t <= 300]
        if len(window) >= 30:
            t0 = window[0][0]
            xs = np.array([t - t0 for t, _ in window])
            ys = np.array([h for _, h in window], dtype=float)
            self.drift_bpm_per_min = float(np.polyfit(xs, ys, 1)[0] * 60.0)
        else:
            self.drift_bpm_per_min = 0.0

    def _update_recovery(self, now):
        window = [h for t, h in self.hr_history if now - t <= 90]
        self.hr_recovery_60s = float(max(window) - self.current_hr) if window else 0.0

    def _accumulate_load(self, dt):
        ratio = max(0.0, (self.current_hr - self.profile.resting_hr) / self.profile.hrr)
        self.training_load += (dt / 60.0) * ratio * 0.64 * math.exp(1.92 * ratio)

    def _detect_phase(self):
        elapsed = self._last_ts - self.session_start
        if elapsed < 240 and self.max_hr_session < self.profile.zone_bounds("Z3")[0]:
            self.phase = Phase.WARMUP
            return
        z = self.zone
        if self.sport == Sport.FOOTBALL:
            self.phase = (Phase.SPRINT if z in ("Z4", "Z5")
                          else Phase.RECOVERY if z in ("Z1", "Z2")
                          else Phase.STEADY)
        else:
            if z in ("Z4", "Z5"): self.phase = Phase.THRESHOLD
            elif z == "Z3":       self.phase = Phase.TEMPO
            elif z == "Z2":       self.phase = Phase.STEADY
            else:
                self.phase = (Phase.RECOVERY
                              if self.avg_hr > self.profile.resting_hr + 30
                              else Phase.COOLDOWN)

    def _track_sprints(self, now):
        if self.sport != Sport.FOOTBALL:
            return
        high = self.zone in ("Z4", "Z5")
        if high and not self._in_sprint:
            self._in_sprint = True
            self._sprint_start_ts = now
            self._sprint_peak_hr = self.current_hr
            self.sprint_count += 1
        elif high and self._in_sprint:
            self._sprint_peak_hr = max(self._sprint_peak_hr, self.current_hr)
        elif not high and self._in_sprint:
            self._in_sprint = False
            self._sprint_end_ts = now
            self._sprint_end_peak = self._sprint_peak_hr
            self._pending_recovery = True

        if self._pending_recovery and now - self._sprint_end_ts >= 30:
            drop = self._sprint_end_peak - self.current_hr
            self.last_sprint_recovery = float(drop)
            self._recovery_samples.append(drop)
            self._pending_recovery = False
            n = len(self._recovery_samples)
            if n >= 3:
                half = n // 2
                early = np.mean(list(self._recovery_samples)[:half])
                late = np.mean(list(self._recovery_samples)[half:])
                self.recovery_trend = ("declining" if late < early - 5
                                       else "improving" if late > early + 5
                                       else "stable")

    # -- cue engine -----------------------------------------------------

    def generate_cue(self) -> Optional[str]:
        if not self.coach_enabled or self.current_hr <= 0:
            return None
        now = time.time()
        if now - self._last_cue_ts < self.min_cue_spacing:
            return None
        ready = [c for c in self._candidates(now)
                 if c.text
                 and c.text != self._last_cue_text
                 and now - self._cue_last.get(c.key, 0.0) >= c.interval]
        if not ready:
            return None
        ready.sort(key=lambda c: c.priority)
        cue = ready[0]
        self._cue_last[cue.key] = now
        self._last_cue_ts = now
        self._last_cue_text = cue.text
        return cue.text

    def _candidates(self, now) -> List[Cue]:
        cues = list(self._safety_cues())
        if self.sport == Sport.FOOTBALL:
            cues += self._football_cues()
        elif self.sport == Sport.HYROX:
            cues += self._hyrox_cues()
        else:
            cues += self._generic_cues()
        cues += self._hrv_cues()
        cues += self._periodic_cues()
        return cues

    def _safety_cues(self):
        pct = (self.current_hr - self.profile.resting_hr) / self.profile.hrr
        if pct >= 0.97 and self.phase != Phase.SPRINT:
            return [Cue(0, "redline", 45,
                f"You're at {self.current_hr}, basically maxed out. "
                f"Back off now and bring it down.")]
        return []

    def _hyrox_cues(self):
        out = []
        lo, hi = self.profile.zone_bounds("Z2")
        elapsed = self._last_ts - self.session_start
        if self.zone in ("Z4", "Z5") and elapsed < 600 and self.phase != Phase.WARMUP:
            out.append(Cue(1, "pace_high", 40,
                f"Too hot for this stage. Settle into zone 2, "
                f"{lo} to {hi} beats, and save it for the back half."))
        if self.zone in ("Z2", "Z3") and self.drift_bpm_per_min > 4:
            out.append(Cue(2, "drift", 90,
                f"Heart rate's drifting up about {round(self.drift_bpm_per_min)} beats a minute "
                f"at the same effort. That's fatigue. Relax your shoulders, deepen the breathing, "
                f"hold your pace."))
        if self.zone == "Z2":
            out.append(Cue(3, "pace_ok", 75,
                "This is your HYROX engine zone. Smooth and controlled, right here."))
        elif self.zone == "Z1" and elapsed > 300:
            out.append(Cue(3, "pace_low", 75,
                "You've got more to give. Lift it into zone 2 for race pace."))
        elif self.zone == "Z3":
            out.append(Cue(3, "pace_tempo", 75,
                "Strong tempo. Fine for training, but on race day hold this for the runs only."))
        return out

    def _football_cues(self):
        out = []
        if self.phase == Phase.SPRINT:
            out.append(Cue(1, "sprint_drive", 18, random.choice([
                "That's the work. Full gas, drive those arms.",
                "Explosive. Stay on your toes and finish the effort.",
                "Top speed now. This is where games are won.",
            ])))
        elif self.phase == Phase.RECOVERY and self.work_time > 30:
            z3_low = self.profile.zone_bounds("Z3")[0]
            if self.hr_recovery_60s >= 25:
                out.append(Cue(1, "recover_good", 35,
                    "Good recovery, heart rate's dropping fast. Reset and get ready to go again."))
            elif self.hr_recovery_60s < 12 and self.current_hr > z3_low:
                out.append(Cue(1, "recover_slow", 35,
                    "Heart rate's staying high between efforts. Walk it off, hands off your knees, "
                    "long slow exhales."))
            else:
                out.append(Cue(2, "recover_ready", 45,
                    "Use this lull. Reposition, scan the play, breathe. Be ready to explode again."))
        if self.recovery_trend == "declining" and self.sprint_count >= 3:
            out.append(Cue(1, "rsa_decline", 90,
                "Your recovery between sprints is slowing down. That's conditioning fatigue. "
                "Pick your moments and go full only when it counts."))
        return out

    def _generic_cues(self):
        out = []
        if self.drift_bpm_per_min > 5 and self.zone in ("Z2", "Z3"):
            out.append(Cue(2, "drift", 90,
                "Heart rate's creeping up at the same effort. Ease back slightly and breathe."))
        out.append(Cue(3, "zone", 75,
            f"Holding zone {self.zone[1]}, {self.ZONE_LABELS.get(self.zone, '')}."))
        return out

    def _hrv_cues(self):
        if self.rmssd > 0 and self.rmssd < 18 and self.phase in (
                Phase.THRESHOLD, Phase.TEMPO, Phase.SPRINT):
            return [Cue(2, "hrv_low", 120,
                "HRV's low, your system's under real load. Lock into a breathing rhythm, "
                "in for three, out for four.")]
        return []

    def _periodic_cues(self):
        mins = int(self._last_ts - self.session_start) // 60
        out = [Cue(4, "status", 120,
                   f"{mins} minute{'s' if mins != 1 else ''} in. "
                   f"Heart rate {self.current_hr}, zone {self.zone[1]}, "
                   f"average {round(self.avg_hr)}.")]
        insight = self._insight_line()
        if insight:
            out.append(Cue(3, "insight", 100, insight))
        return out

    # -- insight + analysis --------------------------------------------

    def _insight_line(self) -> str:
        if self.sport == Sport.FOOTBALL:
            if self.recovery_trend == "declining":
                txt = "Recovery between sprints is slowing. Manage your efforts."
            elif self.work_time > 5 and self.rest_time > 5:
                n = max(1, round(self.rest_time / self.work_time))
                txt = f"Work to rest running about 1 to {n}."
            else:
                txt = "Warming up. Short sharp efforts to prime the engine."
        elif self.sport == Sport.HYROX:
            if self.drift_bpm_per_min > 4 and self.zone in ("Z2", "Z3"):
                txt = "Aerobic drift detected. Hold pace and control breathing."
            else:
                txt = self._readiness()["assessment"]
        else:
            txt = f"Holding zone {self.zone[1]}, {self.ZONE_LABELS.get(self.zone, '')}."
        self.last_insight = txt
        return txt

    def get_coaching_analysis(self) -> dict:
        readiness = self._readiness()
        elapsed = self._last_ts - self.session_start
        breath_rate = self._breathing_rate()
        strain = self._strain_score()
        fatigue = self._fatigue_index()
        autonomic = self._autonomic_balance()
        decoupling = self._cardiac_decoupling()
        recovery_window = self._recovery_window()
        zone_timeline = self._zone_timeline()
        return {
            "sport": self.sport.value,
            "phase": self.phase.value,
            "zone": self.zone,
            "target_zone": self._target_zone(),
            "hr_trend": self.hr_trend,
            "insight": self.last_insight or self._insight_line(),
            "elapsed_seconds": round(elapsed),
            "hrv": {
                "rmssd": round(self.rmssd, 1),
                "sdnn": round(self.sdnn, 1),
                "p50": round(self.pnn50, 1),
                "lf_hf_ratio": round(self.lf_hf, 2),
            },
            "cardiac_drift_bpm_min": round(self.drift_bpm_per_min, 1),
            "hr_recovery_60s": round(self.hr_recovery_60s),
            "recovery_score": self._recovery_score(),
            "training_load": round(self.training_load, 1),
            "strain_score": strain,
            "strain_label": self._strain_label(strain),
            "breathing_rate": breath_rate,
            "fatigue_index": fatigue,
            "fatigue_label": self._fatigue_label(fatigue),
            "autonomic_balance": autonomic,
            "cardiac_decoupling": decoupling,
            "recovery_window_min": recovery_window,
            "zone_timeline": zone_timeline,
            "readiness": readiness,
            "hyrox_readiness": readiness,
            "football": ({
                "sprints": self.sprint_count,
                "work_time_s": round(self.work_time),
                "rest_time_s": round(self.rest_time),
                "recovery_trend": self.recovery_trend,
                "last_sprint_recovery": round(self.last_sprint_recovery),
            } if self.sport == Sport.FOOTBALL else None),
            "suggestions": self._suggestions(),
        }

    def _breathing_rate(self) -> float:
        if len(self.rr_history) < 5:
            return 0.0
        recent = list(self.rr_history)[-20:]
        avg_rr = sum(recent) / len(recent)
        return round(60000.0 / avg_rr * 0.25, 1) if avg_rr > 0 else 0.0

    def _strain_score(self) -> int:
        elapsed = self._last_ts - self.session_start
        if elapsed < 10:
            return 0
        hrs = np.array([h for _, h in self.hr_history], dtype=float)
        if len(hrs) < 10:
            return 0
        pct = (hrs - self.profile.resting_hr) / self.profile.hrr
        pct = np.clip(pct, 0, 1)
        minutes = elapsed / 60.0
        weighted = float(np.mean(pct ** 1.3 * minutes))
        raw = min(21.0, weighted * 2.2)
        return max(0, round(raw))

    def _strain_label(self, score: int) -> str:
        if score <= 9: return "Light"
        if score <= 13: return "Moderate"
        if score <= 17: return "High"
        return "All Out"

    def _fatigue_index(self) -> float:
        if len(self.hr_history) < 60:
            return 0.0
        hrs = np.array([h for _, h in self.hr_history], dtype=float)
        first_half = float(hrs[:len(hrs)//2].mean())
        second_half = float(hrs[len(hrs)//2:].mean())
        drift = second_half - first_half
        hr_factor = drift / max(1, self.profile.hrr) * 100
        hrv_factor = max(0, (30 - self.rmssd) / 30) * 30 if self.rmssd > 0 else 15
        return round(min(100, max(0, hr_factor * 50 + hrv_factor)), 1)

    def _fatigue_label(self, score: float) -> str:
        if score < 25: return "Fresh"
        if score < 50: return "Moderate"
        if score < 75: return "Tired"
        return "Exhausted"

    def _autonomic_balance(self) -> str:
        if self.lf_hf <= 0:
            return "balanced"
        if self.lf_hf > 2.0:
            return "sympathetic"
        if self.lf_hf < 0.5:
            return "parasympathetic"
        return "balanced"

    def _cardiac_decoupling(self) -> float:
        if len(self.hr_history) < 120:
            return 0.0
        window = [(t, h) for t, h in self.hr_history if self._last_ts - t <= 600]
        if len(window) < 60:
            return 0.0
        first_half = [h for _, h in window[:len(window)//2]]
        second_half = [h for _, h in window[len(window)//2:]]
        if not first_half or not second_half:
            return 0.0
        return round(abs(float(np.mean(second_half)) - float(np.mean(first_half))), 1)

    def _recovery_window(self) -> int:
        elapsed = self._last_ts - self.session_start
        if elapsed < 60:
            return 0
        avg_hr_ratio = (self.avg_hr - self.profile.resting_hr) / self.profile.hrr
        drift_factor = abs(self.drift_bpm_per_min) * 2
        base_minutes = elapsed / 60.0 * avg_hr_ratio * 0.5
        return max(5, round(base_minutes + drift_factor))

    def _zone_timeline(self) -> list:
        if not self.hr_history:
            return []
        timeline = []
        current_zone = None
        zone_start = None
        for ts, hr in self.hr_history:
            z = self.profile.zone_for(hr)
            if z != current_zone:
                if current_zone is not None:
                    timeline.append({"zone": current_zone, "start": round(zone_start), "end": round(ts)})
                current_zone = z
                zone_start = ts
        if current_zone:
            timeline.append({"zone": current_zone, "start": round(zone_start), "end": round(self._last_ts)})
        return timeline[-20:]

    def calculate_hrv_metrics(self) -> dict:
        return self.get_coaching_analysis()["hrv"]

    def _target_zone(self) -> str:
        if self.sport == Sport.FOOTBALL:
            return "Z2"
        return {Phase.WARMUP: "Z1", Phase.STEADY: "Z2", Phase.TEMPO: "Z3",
                Phase.THRESHOLD: "Z4", Phase.RECOVERY: "Z1",
                Phase.COOLDOWN: "Z1"}.get(self.phase, "Z2")

    def _readiness(self) -> dict:
        total = sum(self.zone_time.values())
        if total < 30:
            return {"score": 0, "assessment": "Gathering data..."}
        if self.sport == Sport.FOOTBALL:
            base = 60
            if self.recovery_trend == "improving": base += 15
            elif self.recovery_trend == "declining": base -= 20
            if self.last_sprint_recovery >= 25: base += 10
            score = max(0, min(100, base))
            if score >= 75: a = "Sharp. Repeat-sprint ability holding up well."
            elif score >= 55: a = "Solid intensity. Watch recovery as fatigue builds."
            else: a = "Fatiguing. Recovery between efforts is dropping off."
            return {"score": score, "assessment": a}
        z2 = self.zone_time["Z2"]
        hi = self.zone_time["Z4"] + self.zone_time["Z5"]
        z2_pct, hi_pct = z2 / total, hi / total
        if z2_pct > 0.6 and hi_pct < 0.15:
            return {"score": 85, "assessment": "Excellent aerobic base for HYROX."}
        if z2_pct > 0.4:
            return {"score": 70, "assessment": "Good aerobic fitness. Build more Z2 volume."}
        if hi_pct > 0.3:
            return {"score": 50, "assessment": "Too much high intensity for an aerobic session."}
        return {"score": 60, "assessment": "Building fitness. Favor consistent Z2."}

    def _recovery_score(self) -> int:
        score = 50
        if self.rmssd > 0: score += min(30, int(self.rmssd / 2))
        if self.hr_trend == "falling": score += 10
        elif self.hr_trend == "rising": score -= 10
        if self.drift_bpm_per_min > 5: score -= 10
        return max(0, min(100, score))

    def _suggestions(self) -> list:
        s = []
        elapsed = self._last_ts - self.session_start
        if self.rmssd and self.rmssd < 25 and elapsed > 300:
            s.append("HRV is low. Consider easing intensity or a recovery day.")
        total = sum(self.zone_time.values())
        if self.sport == Sport.HYROX and total > 0:
            if self.zone_time["Z1"] / total > 0.5 and elapsed > 600:
                s.append("Lots of Z1. For HYROX, aim for more Z2 volume.")
            if self.drift_bpm_per_min > 6:
                s.append("Significant cardiac drift. Hydrate and check pacing.")
        if self.sport == Sport.FOOTBALL and self.recovery_trend == "declining":
            s.append("Recovery slowing between sprints. Rest fully in low-intensity phases.")
        return s

    # -- TTS ------------------------------------------------------------

    async def speak(self, text: str) -> Optional[str]:
        if not text:
            return None
        try:
            communicate = edge_tts.Communicate(text, self.tts_voice)
            audio = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
            return base64.b64encode(audio).decode("utf-8")
        except Exception as e:
            print(f"[coach] TTS error: {e}")
            return None


# Back-compat
HYROXCoach = LiveCoach
