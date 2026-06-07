# HYROX Coach

Real-time heart-rate coaching for HYROX training. Polar H10 chest strap → BLE → Python server → live dashboard on iPhone → voice coaching through AirPods.

## What It Does

- **Live HR streaming** from Polar H10 via BLE (using bleakheart)
- **Multi-sport coaching engine**: HYROX, Football, Generic Cardio
- **Real-time analysis**: HR zones, HRV (RMSSD, SDNN, pNN50, LF/HF), cardiac drift, HR recovery, repeat-sprint ability
- **TTS voice coaching** via `edge-tts` (different voices per sport) — audio routes to AirPods or Meta glasses
- **PWA dashboard** — runs on iPhone Safari, full-screen, no app store needed
- **Session history** — SQLite storage, pattern tracking over time

## Quick Start

### Desktop Shortcut (Easiest)
Double-click **`HYROX Coach.bat`** on your Desktop. It will:
1. Kill any old Python processes
2. Start the server on port **8765**

### Manual
```bash
cd hyrox-coach-final
start.bat
```

Or directly:
```bash
pip install -r requirements.txt
python scripts/server.py
```

### Open on iPhone
Navigate to `http://<your-pc-ip>:8765` (the server prints the exact URL at startup).

## Setup Requirements

1. **Python 3.11** at `%LOCALAPPDATA%\Programs\Python\Python311\python.exe`
2. **Dependencies** auto-installed by `start.bat`:
   - `bleak` — BLE communication
   - `bleakheart` — Polar H10 specific parsing (HR, RR intervals, ECG, battery, skin-contact)
   - `aiohttp` — WebSocket + HTTP server
   - `edge-tts` — Microsoft neural TTS for coaching voice
   - `numpy` — HRV analysis

## Connection Modes

### 1. Server BLE Relay (preferred)
- Server reads H10 directly via bleakheart
- H10 must NOT be connected to your iPhone Bluetooth
- Auto-reconnects on drop
- Shows battery level, skin-contact, streaming status

### 2. Web Bluetooth (fallback)
- iPhone browser connects to H10 directly
- Data forwarded to server via WebSocket
- Desktop Chrome/Edge only (not supported on iOS Safari)

## Sports & Coaching

| Sport | Focus | Voice |
|-------|-------|-------|
| **HYROX** | Compromised-running pacing, aerobic discipline, Z2 base building | en-US-GuyNeural (calm) |
| **Football** | Sprint/recovery cues, repeat-sprint ability, fatigue management | en-US-ChristopherNeural (punchy) |
| **Generic** | Zone-based coaching for any cardio session | en-US-GuyNeural |

### Cue Intelligence
- **Priorities**: safety > sport action > HRV > pacing > status
- **Cooldowns**: each cue type has a min interval (e.g. redline alert = 45s, status update = 2min)
- **Non-repetitive**: never speaks the same cue twice in a row
- **Sport-aware**: football never tells you to slow down mid-sprint

### Key Cues
- *Cardiac drift*: "Heart rate's drifting up 4 beats a minute at the same effort. That's fatigue. Hold your pace."
- *HR recovery* (football): "Good recovery, heart rate's dropping fast. Reset and get ready to go again."
- *Repeat-sprint decline* (football): "Your recovery between sprints is slowing. Pick your moments."
- *HRV low*: "Lock into a breathing rhythm, in for three, out for four."

## Architecture

```
Polar H10 ──BLE──→ Python Server (bleakheart) ──WebSocket──→ iPhone Safari (PWA)
       │                                                    │
       └── Web Bluetooth (browser direct) ──────────────────┘

Server broadcasts:
  • HR data (1Hz)         → live HR display, charts
  • Coaching analysis     → phase, zones, insights
  • TTS audio (MP3/base64) → plays through AirPods
  • Session events        → start/stop, history
```

## File Structure

```
hyrox-coach-final/
├── scripts/
│   ├── server.py       # Main server: WebSocket, HTTP, coaching, TTS, SQLite
│   ├── ble_relay.py    # Polar H10 BLE relay using bleakheart
│   └── coach.py        # LiveCoach multi-sport engine
├── web/
│   ├── index.html      # PWA dashboard (vanilla JS, Canvas charts)
│   └── manifest.json   # PWA manifest
├── requirements.txt    # Python dependencies
├── start.bat           # Windows startup script
├── sessions.db         # SQLite (auto-created, gitignored)
└── README.md
```

## Configuration (in Dashboard Settings tab)

- **Max Heart Rate** — your actual max HR (default 190)
- **Resting Heart Rate** — your resting HR (default 60)
- **Sport** — HYROX / Football / Generic
- **Coach Voice Interval** — 15s / 30s (default) / 60s / 2min
- **Voice Coaching** — toggle on/off

## Known Issues

1. **H10 connects to one device at a time**: If paired with iPhone, server BLE relay can't see it. Use Web Bluetooth or disconnect from iPhone settings.
2. **Web Bluetooth not supported on iOS Safari**: Use the server's BLE relay instead.
3. **Port 8765**: Make sure no other app is using it. The `start.bat` kills old Python processes to free the port.

## Changelog

- **v2.1** — Bug fixes: stop session auto-restart, iOS TTS audio, port standardisation
- **v2.0** — Multi-sport coaching, bleakheart relay, cardiac drift, HR recovery
- **v1.0** — Initial release

## License

Personal project. No warranty.
