"""
coach.py - Multi-sport live coaching engine v3.0.

Turns a stream of heart-rate / RR samples into:
  * meaningful real-time analysis (zones, HRV, cardiac drift, HR recovery,
    training load, sport-specific readiness), and
  * spoken coaching cues (TTS) that are specific, prioritised, varied, and
    genuinely insightful.

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
  coach.generate_session_summary() -> dict for post-session review
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
    STEADY = "steady"
    TEMPO = "tempo"
    THRESHOLD = "threshold"
    SPRINT = "sprint"
    RECOVERY = "recovery"
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
        self._phase_transition_ts: Dict[str, float] = {}
        self._breathing_cue_count = 0
        self._motivation_count = 0

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
        cues += self._breathing_cues()
        cues += self._fatigue_cues()
        cues += self._motivation_cues()
        cues += self._periodic_cues()
        return cues

    def _safety_cues(self):
        pct = (self.current_hr - self.profile.resting_hr) / self.profile.hrr
        if pct >= 0.97 and self.phase != Phase.SPRINT:
            return [Cue(0, "redline", 45,
                random.choice([
                    f"You're at {self.current_hr}, basically at your ceiling. Ease off now, bring it down gradually.",
                    f"Heart rate at {self.current_hr}, that's redline territory. Back off, long exhales, let it settle.",
                    f"You're pushing past safe limits at {self.current_hr}. Dial it back, focus on controlled breathing.",
                ]))]
        if pct >= 0.93 and self.phase != Phase.SPRINT:
            return [Cue(0, "near_redline", 60,
                random.choice([
                    f"Heart rate at {self.current_hr}, you're approaching your ceiling. Stay controlled.",
                    f"Almost at your max. {self.current_hr} is close to the edge. Hold steady, don't spike it.",
                ]))]
        return []

    def _hyrox_cues(self):
        out = []
        elapsed = self._last_ts - self.session_start
        lo, hi = self.profile.zone_bounds("Z2")
        z3_lo, z3_hi = self.profile.zone_bounds("Z3")
        z4_lo = self.profile.zone_bounds("Z4")[0]

        # --- Pacing cues ---
        if self.zone in ("Z4", "Z5") and elapsed < 600 and self.phase != Phase.WARMUP:
            out.append(Cue(1, "pace_high", 40,
                random.choice([
                    f"Too hot this early. Settle into zone 2, {lo} to {hi} beats, and save it for the back half.",
                    f"You're burning matches. Drop to {lo}-{hi} BPM, this is a long game.",
                    f"Easy now. You should be in the {lo}-{hi} range for the first 10 minutes. Control it.",
                    f"That's threshold work and you're not even 10 minutes in. Pull back to {lo}-{hi}.",
                ])))

        if self.zone in ("Z2", "Z3") and self.drift_bpm_per_min > 4:
            out.append(Cue(2, "drift", 90,
                random.choice([
                    f"Heart rate's drifting up about {round(self.drift_bpm_per_min)} beats a minute at the same effort. That's fatigue setting in. Relax your shoulders, deepen the breathing, hold your pace.",
                    f"Cardiac drift detected: plus {round(self.drift_bpm_per_min)} BPM over 5 minutes at the same effort. Slow your breathing, stay smooth.",
                    f"Your heart rate's creeping up even though you haven't sped up. That's aerobic drift. Ease your tension, breathe low and slow.",
                    f"Drift of {round(self.drift_bpm_per_min)} BPM per minute. Your body's working harder for the same pace. Focus on staying relaxed.",
                ])))

        # --- Zone-specific encouragement ---
        if self.zone == "Z2":
            out.append(Cue(3, "pace_ok", 75,
                random.choice([
                    "This is your HYROX engine zone. Smooth and controlled, right here.",
                    "Perfect pacing. Z2 is where aerobic fitness is built. Lock it in.",
                    "You're in the sweet spot. Steady, rhythmic, sustainable. This is race pace.",
                    "Zone 2. This is where the magic happens. Patient, controlled, powerful.",
                    "Right where you need to be. Let your body settle into this rhythm.",
                ])))
        elif self.zone == "Z1" and elapsed > 300:
            out.append(Cue(3, "pace_low", 75,
                random.choice([
                    "You've got more to give. Lift it into zone 2 for race pace.",
                    "Too easy. Push the pace slightly, get into the {lo}-{hi} range.",
                    "You're coasting. Bring it up to zone 2, that's where the training effect is.",
                    "Drop too low and you lose the training stimulus. Nudge it up a bit.",
                ])))
        elif self.zone == "Z3":
            out.append(Cue(3, "pace_tempo", 75,
                random.choice([
                    "Strong tempo. Fine for intervals, but on race day hold this for the runs only.",
                    "Zone 3, that's tempo territory. Good for push segments, but be careful not to drift higher.",
                    "Solid effort, but you're in no-man's land. Either drop to Z2 for steady or commit to Z4 for intervals.",
                ])))
        elif self.zone == "Z4":
            out.append(Cue(3, "pace_threshold", 75,
                random.choice([
                    "Threshold work. You can hold this for a few minutes, but not forever. Breathe through it.",
                    "That's threshold pace. Strong effort. Focus on your form, keep your cadence up.",
                    "Z4, that's race-day push pace. Controlled aggression. Don't let your form break down.",
                ])))
        elif self.zone == "Z5":
            out.append(Cue(3, "pace_max", 60,
                random.choice([
                    "Maximum effort. This is where you build top-end speed. Drive those arms.",
                    "All out. Short bursts of Z5 build power. Stay efficient, don't tense up.",
                    "Full gas. Keep it short and sharp. Your body needs to recover from this.",
                ])))

        # --- Running form cues ---
        if self.zone in ("Z2", "Z3") and elapsed > 180:
            out.append(Cue(3, "form", 150,
                random.choice([
                    "Quick form check. Shoulders down, hands relaxed, slight forward lean. Let gravity help.",
                    "Focus on your cadence. Short, quick steps. Don't overstride.",
                    "Belly breathing. Let your diaphragm do the work, not your chest.",
                    "Check your hands. Are they clenched? Shake them out, keep them loose.",
                    "Head up, eyes forward. Don't look at your feet, look where you're going.",
                    "Relax your jaw. Tension in your face means tension in your shoulders.",
                ])))

        # --- Breathing cues ---
        if elapsed > 120 and self._breathing_cue_count < 3:
            techniques = [
                "Try nasal breathing for the next minute. In through the nose, out through the nose. It trains your aerobic system.",
                "Box breathing: in for 4, hold for 4, out for 4, hold for 4. Even one cycle resets your focus.",
                "Sync your breath to your stride. Two steps per inhale, two steps per exhale. It creates rhythm.",
                "Exhale fully. Most people under-breathe. A complete exhale lets a deeper inhale happen naturally.",
                "Breathe low, not high. Diaphragmatic breathing delivers more oxygen with less effort.",
            ]
            out.append(Cue(3, "breathing_tech", 120, random.choice(techniques)))
            self._breathing_cue_count += 1

        # --- Mental coaching ---
        if elapsed > 300 and elapsed < 1200:
            out.append(Cue(4, "mental", 180,
                random.choice([
                    "This is the middle stretch. The part where most people quit mentally. Stay present.",
                    "Don't think about how much is left. Just this minute. This stride. Right now.",
                    "The discomfort you feel is your body adapting. Embrace it.",
                    "Find a mantra. Something short. Repeat it. It anchors your focus.",
                    "Visualize the finish line. How do you want to feel when this is done?",
                ])))

        return out

    def _football_cues(self):
        out = []
        if self.phase == Phase.SPRINT:
            out.append(Cue(1, "sprint_drive", 18, random.choice([
                "That's the work. Full gas, drive those arms.",
                "Explosive. Stay on your toes and finish the effort.",
                "Top speed now. This is where games are won.",
                "Go go go. Don't decelerate until you hear the whistle.",
                "Power through. Every stride counts. Leave nothing.",
                "Attack it. Fast feet, strong arms, full commitment.",
                "Sprint mechanics: high knees, quick turnover, lean forward slightly.",
            ])))
        elif self.phase == Phase.RECOVERY and self.work_time > 30:
            z3_low = self.profile.zone_bounds("Z3")[0]
            if self.hr_recovery_60s >= 25:
                out.append(Cue(1, "recover_good", 35,
                    random.choice([
                        "Good recovery, heart rate's dropping fast. Reset and get ready to go again.",
                        "That's a solid drop. Your body's handling the load well. Stay composed.",
                        "Strong recovery. You're bouncing back quick. Use this to recharge.",
                    ])))
            elif self.hr_recovery_60s < 12 and self.current_hr > z3_low:
                out.append(Cue(1, "recover_slow", 35,
                    random.choice([
                        "Heart rate's staying high between efforts. Walk it off, hands off your knees, long slow exhales.",
                        "Recovery's slow. Drop your arms to your sides, breathe through your nose, walk slowly.",
                        "Your heart rate's not dropping. Stop running, walk, hands on head, deep breaths.",
                    ])))
            else:
                out.append(Cue(2, "recover_ready", 45,
                    random.choice([
                        "Use this lull. Reposition, scan the play, breathe. Be ready to explode again.",
                        "Active recovery. Light jog, controlled breathing. Stay ready.",
                        "Reset. This is where you gain an advantage. Stay alert.",
                        "Easy movement. Let your heart rate come down, but keep your mind sharp.",
                    ])))

        if self.recovery_trend == "declining" and self.sprint_count >= 3:
            out.append(Cue(1, "rsa_decline", 90,
                random.choice([
                    "Your recovery between sprints is slowing down. That's conditioning fatigue. Pick your moments.",
                    "Recovery's getting harder. Be selective with your efforts, go full when it matters.",
                    "Your body's telling you it's fatiguing. Smart effort selection wins games.",
                ])))

        # Sprint-specific form cues
        if self.zone in ("Z4", "Z5") and self._in_sprint:
            out.append(Cue(2, "sprint_form", 25,
                random.choice([
                    "Drive your arms. Arm speed equals leg speed.",
                    "Stay on the balls of your feet. Heels barely touch the ground.",
                    "Relax your face. Tension there slows you down.",
                    "Lean forward slightly. Let gravity pull you.",
                ])))

        return out

    def _generic_cues(self):
        out = []
        elapsed = self._last_ts - self.session_start
        lo, hi = self.profile.zone_bounds("Z2")

        if self.drift_bpm_per_min > 5 and self.zone in ("Z2", "Z3"):
            out.append(Cue(2, "drift", 90,
                random.choice([
                    "Heart rate's creeping up at the same effort. Ease back slightly and breathe.",
                    "Cardiac drift detected. Your body's working harder for the same output. Dial it back a touch.",
                    "HR is drifting upward. Drop the intensity a notch and focus on breathing.",
                ])))

        if self.zone == "Z2":
            out.append(Cue(3, "zone", 75,
                random.choice([
                    f"Holding zone 2, aerobic base. This is where endurance is built.",
                    f"Steady aerobic work. The foundation of all fitness. Keep it smooth.",
                    f"Zone 2, that's the sweet spot. Patient and controlled.",
                ])))
        elif self.zone == "Z3":
            out.append(Cue(3, "zone", 75,
                random.choice([
                    f"Zone 3, tempo. A good hard effort. Monitor how you feel.",
                    f"Tempo zone. Pushing the envelope but sustainable. For now.",
                ])))
        elif self.zone in ("Z4", "Z5"):
            out.append(Cue(3, "zone", 60,
                random.choice([
                    f"High intensity, zone {self.zone[1]}. This is where adaptations happen fast.",
                    f"Threshold or above. Short bursts are fine, but don't linger here too long.",
                ])))
        elif self.zone == "Z1":
            out.append(Cue(3, "zone", 90,
                random.choice([
                    f"Easy zone. Good for warm-up or cool-down. Recovery mode.",
                    f"Zone 1, recovery. Let your body absorb the work.",
                ])))

        return out

    def _hrv_cues(self):
        if self.rmssd > 0 and self.rmssd < 18 and self.phase in (
                Phase.THRESHOLD, Phase.TEMPO, Phase.SPRINT):
            return [Cue(2, "hrv_low", 120,
                random.choice([
                    "HRV's low, your system's under real load. Lock into a breathing rhythm, in for three, out for four.",
                    "Heart rate variability is dropping. Your nervous system is stressed. Slow your breathing, stay calm.",
                    "HRV indicates high strain. Take a moment. Inhale for 4 counts, exhale for 6.",
                    "Low HRV means your body is fighting hard. Ease the intensity if you can, breathe deep.",
                ]))]
        if self.rmssd > 40 and self.phase in (Phase.STEADY, Phase.RECOVERY):
            return [Cue(4, "hrv_good", 180,
                random.choice([
                    "HRV looks strong. Your body is handling this well. Keep going.",
                    "Good heart rate variability. You're recovering nicely between beats.",
                ]))]
        return []

    def _breathing_cues(self):
        """Cue breathing technique and rhythm at appropriate times."""
        elapsed = self._last_ts - self.session_start
        if elapsed < 60:
            return []
        cues = []
        if self.zone in ("Z3", "Z4") and self._breathing_cue_count < 5:
            cues.append(Cue(3, "breath_rhythm", 100,
                random.choice([
                    "Match your breathing to your steps. Two in, two out. Find your rhythm.",
                    "Long exhales. Make the exhale longer than the inhale. It activates your parasympathetic system.",
                    "Breathe from your belly, not your chest. Low, slow, deep.",
                    "Try a 3:2 breathing pattern. Three steps inhale, two steps exhale. It stabilizes your core.",
                ])))
            self._breathing_cue_count += 1
        if self.zone == "Z5":
            cues.append(Cue(2, "breath_sprint", 30,
                random.choice([
                    "Don't hold your breath. Exhale hard on every stride.",
                    "Power breathing. Forceful exhales, let the inhale happen naturally.",
                ])))
        return cues

    def _fatigue_cues(self):
        """Detect fatigue patterns and offer targeted advice."""
        elapsed = self._last_ts - self.session_start
        if elapsed < 180:
            return []
        cues = []
        fi = self._fatigue_index()
        if fi > 70 and self.phase != Phase.SPRINT:
            cues.append(Cue(2, "fatigue_high", 90,
                random.choice([
                    "Fatigue index is high. Your form is probably slipping. Focus on posture: tall, relaxed, rhythmic.",
                    "You're fatiguing fast. Slow down a touch and focus on running economy. Wasted energy compounds.",
                    "Your body is working overtime. Drop the intensity and focus on efficiency over speed.",
                ])))
        elif fi > 50 and self.phase != Phase.SPRINT:
            cues.append(Cue(3, "fatigue_moderate", 120,
                random.choice([
                    "Moderate fatigue building. This is normal. Stay mentally engaged, don't let your form deteriorate.",
                    "Fatigue is accumulating. Focus on staying relaxed. Tension wastes energy.",
                ])))

        # Cardiac decoupling
        decoupling = self._cardiac_decoupling()
        if decoupling > 5 and elapsed > 300:
            cues.append(Cue(3, "decoupling", 120,
                random.choice([
                    f"Cardiac decoupling at {decoupling} BPM. Your heart rate and effort are diverging. Slow down, hydrate, refocus.",
                    f"HR is {decoupling} BPM higher than it should be for this effort. Your body needs a break. Ease back.",
                ])))

        return cues

    def _motivation_cues(self):
        """Provide contextual motivation based on session progress."""
        elapsed = self._last_ts - self.session_start
        mins = elapsed / 60.0
        cues = []

        if mins > 0 and mins % 10 < 0.1 and self._motivation_count < 4:
            if mins <= 10:
                cues.append(Cue(4, "motivation_early", 300,
                    random.choice([
                        "10 minutes in. You showed up. That's the hardest part.",
                        "First 10 done. The hardest part is over. Now you're in the zone.",
                    ])))
            elif mins <= 25:
                cues.append(Cue(4, "motivation_mid", 300,
                    random.choice([
                        "Halfway through if this is a 30. You're in a great rhythm.",
                        "20 minutes in. Strong and steady. This is where fitness is earned.",
                        "Deep into the session. Your body is adapting. Keep pushing.",
                    ])))
            elif mins <= 40:
                cues.append(Cue(4, "motivation_late", 300,
                    random.choice([
                        "30 minutes. You've put in serious work today. Finish strong.",
                        "Over 30 minutes of solid training. You're building something real.",
                    ])))
            else:
                cues.append(Cue(4, "motivation_end", 300,
                    random.choice([
                        "40 plus minutes. You're a machine. This is elite-level consistency.",
                        "Over 40 minutes. Most people quit at 20. You're different.",
                    ])))
            self._motivation_count += 1

        return cues

    def _periodic_cues(self):
        mins = int(self._last_ts - self.session_start) // 60
        elapsed = self._last_ts - self.session_start
        out = []

        # Rich status update
        if elapsed > 60:
            lo, hi = self.profile.zone_bounds(self.zone)
            zt_min = int(self.zone_time.get(self.zone, 0)) // 60
            out.append(Cue(4, "status", 120,
                f"{mins} minute{'s' if mins != 1 else ''} in. "
                f"Heart rate {self.current_hr}, zone {self.zone[1]}, "
                f"average {round(self.avg_hr)}. "
                f"You've been in zone {self.zone[1]} for {zt_min} minutes."))

        insight = self._insight_line()
        if insight:
            out.append(Cue(3, "insight", 100, insight))

        # Hydration reminder
        if mins > 0 and mins % 20 == 0 and mins > 0:
            out.append(Cue(4, "hydrate", 600,
                random.choice([
                    "Quick reminder: sip some water. Even a small amount helps performance.",
                    "Hydration check. Take a quick drink. Dehydration kills performance.",
                    "Drink something. Your body needs fluid. Even a few sips make a difference.",
                ])))

        return out

    # -- insight + analysis --------------------------------------------

    def _insight_line(self) -> str:
        if self.sport == Sport.FOOTBALL:
            if self.recovery_trend == "declining":
                txt = random.choice([
                    "Recovery between sprints is slowing. Manage your efforts.",
                    "Your sprint recovery is declining. Be smarter about when you go all-out.",
                ])
            elif self.work_time > 5 and self.rest_time > 5:
                n = max(1, round(self.rest_time / self.work_time))
                txt = f"Work to rest ratio about 1 to {n}."
            else:
                txt = "Warming up. Short sharp efforts to prime the engine."
        elif self.sport == Sport.HYROX:
            if self.drift_bpm_per_min > 4 and self.zone in ("Z2", "Z3"):
                txt = random.choice([
                    "Aerobic drift detected. Hold pace and control breathing.",
                    "Your heart rate is drifting. Stay relaxed, don't let tension pull it higher.",
                ])
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

    # -- session summary ------------------------------------------------

    def generate_session_summary(self) -> dict:
        """Generate a comprehensive post-session summary with insights."""
        elapsed = self._last_ts - self.session_start
        hrs = [h for _, h in self.hr_history]
        if not hrs:
            return {"error": "No data collected"}

        duration_mins = elapsed / 60.0
        total_zone_time = sum(self.zone_time.values())
        zone_pcts = {}
        for z in ("Z1", "Z2", "Z3", "Z4", "Z5"):
            zone_pcts[z] = round(self.zone_time[z] / total_zone_time * 100, 1) if total_zone_time > 0 else 0

        # Determine session grade
        grade = self._session_grade()

        # Time in zones (formatted)
        zone_breakdown = {}
        for z in ("Z1", "Z2", "Z3", "Z4", "Z5"):
            secs = self.zone_time[z]
            zone_breakdown[z] = {
                "seconds": round(secs),
                "minutes": round(secs / 60, 1),
                "percent": zone_pcts[z],
            }

        # Key insights
        insights = self._session_insights(duration_mins, zone_pcts)

        # Calorie estimate
        avg_hr_for_calcs = self.avg_hr
        calories = round(duration_mins * (avg_hr_for_calcs * 0.0035 + (0.0175 * max(0, avg_hr_for_calcs - 120)) + 0.175))

        # HR peaks and valleys
        if hrs:
            # Find peaks (local maxima > 160)
            peaks = []
            for i in range(2, len(hrs) - 2):
                if hrs[i] > hrs[i-1] and hrs[i] > hrs[i-2] and hrs[i] > hrs[i+1] and hrs[i] > hrs[i+2] and hrs[i] > 150:
                    peaks.append(hrs[i])
            peak_info = f"{len(peaks)} peaks detected" if peaks else "No significant peaks"
            peak_max = max(peaks) if peaks else 0
        else:
            peak_info = "No peaks"
            peak_max = 0

        # Training effect
        training_effect = self._training_effect_label()

        return {
            "duration_seconds": round(elapsed),
            "duration_mins": round(duration_mins, 1),
            "avg_hr": round(self.avg_hr),
            "max_hr": self.max_hr_session,
            "min_hr": self.min_hr_session if self.min_hr_session < 999 else 0,
            "calories": calories,
            "strain_score": self._strain_score(),
            "strain_label": self._strain_label(self._strain_score()),
            "training_load": round(self.training_load, 1),
            "training_effect": training_effect,
            "grade": grade,
            "zone_breakdown": zone_breakdown,
            "hrv": {
                "rmssd": round(self.rmssd, 1),
                "sdnn": round(self.sdnn, 1),
                "pnn50": round(self.pnn50, 1),
                "lf_hf": round(self.lf_hf, 2),
            },
            "cardiac_drift": round(self.drift_bpm_per_min, 1),
            "hr_recovery_60s": round(self.hr_recovery_60s),
            "fatigue_index": round(self._fatigue_index(), 1),
            "fatigue_label": self._fatigue_label(self._fatigue_index()),
            "autonomic_balance": self._autonomic_balance(),
            "readiness": self._readiness(),
            "recovery_window_min": self._recovery_window(),
            "insights": insights,
            "suggestions": self._suggestions(),
            "peak_info": peak_info,
            "peak_max_hr": peak_max,
            "sport": self.sport.value,
            "football": ({
                "total_sprints": self.sprint_count,
                "total_work_time_s": round(self.work_time),
                "total_rest_time_s": round(self.rest_time),
                "recovery_trend": self.recovery_trend,
                "avg_sprint_recovery": round(np.mean(list(self._recovery_samples)), 1) if self._recovery_samples else 0,
            } if self.sport == Sport.FOOTBALL else None),
        }

    def _session_grade(self) -> dict:
        """Calculate an overall session grade based on multiple factors."""
        elapsed = self._last_ts - self.session_start
        duration_mins = elapsed / 60.0
        total_zone_time = sum(self.zone_time.values())

        score = 50  # Base score

        # Duration bonus
        if duration_mins >= 45:
            score += 15
        elif duration_mins >= 30:
            score += 10
        elif duration_mins >= 20:
            score += 5

        # Zone discipline (HYROX)
        if self.sport == Sport.HYROX and total_zone_time > 0:
            z2_pct = self.zone_time["Z2"] / total_zone_time
            if z2_pct > 0.6:
                score += 15  # Great Z2 discipline
            elif z2_pct > 0.4:
                score += 8
            hi_pct = (self.zone_time["Z4"] + self.zone_time["Z5"]) / total_zone_time
            if hi_pct < 0.15:
                score += 5
            elif hi_pct > 0.3:
                score -= 10  # Too much intensity

        # HRV quality
        if self.rmssd > 40:
            score += 10
        elif self.rmssd > 25:
            score += 5
        elif self.rmssd < 15:
            score -= 5

        # Low cardiac drift is good
        if abs(self.drift_bpm_per_min) < 3:
            score += 5
        elif self.drift_bpm_per_min > 6:
            score -= 5

        # Fatigue management
        fi = self._fatigue_index()
        if fi < 40:
            score += 5
        elif fi > 70:
            score -= 5

        score = max(0, min(100, score))

        if score >= 85:
            letter, label, color = "A", "Excellent Session", "#22c55e"
        elif score >= 75:
            letter, label, color = "B+", "Strong Session", "#84cc16"
        elif score >= 65:
            letter, label, color = "B", "Good Session", "#84cc16"
        elif score >= 55:
            letter, label, color = "C+", "Decent Session", "#eab308"
        elif score >= 45:
            letter, label, color = "C", "Average Session", "#eab308"
        elif score >= 35:
            letter, label, color = "D", "Below Average", "#f97316"
        else:
            letter, label, color = "F", "Needs Improvement", "#ef4444"

        return {"letter": letter, "label": label, "score": score, "color": color}

    def _training_effect_label(self) -> str:
        """Estimate training effect based on load and zones."""
        elapsed = self._last_ts - self.session_start
        duration_mins = elapsed / 60.0
        total_zone_time = sum(self.zone_time.values())
        if total_zone_time <= 0:
            return "Minimal"

        z5_pct = self.zone_time["Z5"] / total_zone_time
        z4_pct = self.zone_time["Z4"] / total_zone_time
        z2_pct = self.zone_time["Z2"] / total_zone_time

        if self.training_load > 300:
            return "Overreaching"
        elif self.training_load > 200 or z5_pct > 0.1:
            return "VO2 Max"
        elif self.training_load > 120 or z4_pct > 0.15:
            return "Threshold"
        elif self.training_load > 60 and z2_pct > 0.4:
            return "Aerobic Base"
        elif duration_mins > 20:
            return "Recovery"
        else:
            return "Warm-up"

    def _session_insights(self, duration_mins: float, zone_pcts: dict) -> list:
        """Generate textual insights about the session."""
        insights = []

        # Duration insight
        if duration_mins >= 60:
            insights.append(f"Outstanding {round(duration_mins)}-minute session. Long endurance work like this builds serious aerobic capacity.")
        elif duration_mins >= 45:
            insights.append(f"Solid {round(duration_mins)}-minute session. Great volume for building fitness.")
        elif duration_mins >= 30:
            insights.append(f"{round(duration_mins)}-minute session. Good training stimulus for the day.")
        elif duration_mins >= 15:
            insights.append(f"Short {round(duration_mins)}-minute session. Every minute counts, but aim for 30+ next time.")
        else:
            insights.append(f"Quick {round(duration_mins)}-minute session. Better than nothing, but try to extend it next time.")

        # Zone discipline
        if self.sport == Sport.HYROX:
            if zone_pcts.get("Z2", 0) > 60:
                insights.append("Excellent Z2 discipline. This is exactly how HYROX runners build their aerobic engine.")
            elif zone_pcts.get("Z2", 0) > 40:
                insights.append("Good amount of Z2 work. Try to keep even more time in this zone for optimal HYROX training.")
            if zone_pcts.get("Z4", 0) + zone_pcts.get("Z5", 0) > 25:
                insights.append("High-intensity zones dominated. For HYROX race prep, keep intensity lower and build volume.")
        elif self.sport == Sport.FOOTBALL:
            insights.append(f"Completed {self.sprint_count} high-intensity efforts. Work-to-rest ratio: {round(self.work_time)}s work, {round(self.rest_time)}s rest.")

        # HRV insight
        if self.rmssd > 40:
            insights.append("Strong HRV throughout. Your autonomic nervous system handled the load well.")
        elif self.rmssd > 25:
            insights.append("Moderate HRV. Your body handled the session but was working for it.")
        elif self.rmssd > 0:
            insights.append("Low HRV indicates significant stress. Consider a recovery day before your next hard session.")

        # Cardiac drift
        if abs(self.drift_bpm_per_min) > 6:
            insights.append(f"Notable cardiac drift ({round(self.drift_bpm_per_min)} BPM/min). Your cardiovascular system was under significant strain. Hydration and pacing could help.")
        elif abs(self.drift_bpm_per_min) < 2 and duration_mins > 15:
            insights.append("Minimal cardiac drift. Excellent cardiovascular control throughout the session.")

        # Fatigue
        fi = self._fatigue_index()
        if fi > 70:
            insights.append("High fatigue accumulation. Your body was pushed hard today. Prioritize recovery before your next intense session.")
        elif fi > 50:
            insights.append("Moderate fatigue. Normal for a training day. Listen to your body for recovery signals.")

        # Recovery
        if self.hr_recovery_60s > 30:
            insights.append("Fast HR recovery between efforts. Your cardiovascular fitness is improving.")
        elif self.hr_recovery_60s < 10 and duration_mins > 5:
            insights.append("Slow heart rate recovery. Your body is still adapting to this level of intensity.")

        # Overall assessment
        if self._strain_score() >= 17:
            insights.append("This was an all-out session. You gave everything. Make sure tomorrow is easy or a rest day.")
        elif self._strain_score() >= 13:
            insights.append("High-strain session. Effective training, but ensure adequate sleep and nutrition for recovery.")

        return insights

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
