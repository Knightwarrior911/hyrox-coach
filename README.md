# HYROX Coach — Real-Time Heart Rate Coaching

AI-powered running coach that listens to your Polar H10, analyzes your heart rate in real-time, and talks to you through your AirPods.

## What It Does

- **Live HR streaming** from Polar H10 chest strap via BLE
- **Real-time coaching** via TTS (text-to-speech) — audio cues in your AirPods
- **HRV analysis** — RMSSD, SDNN, LF/HF ratio, frequency domain
- **HYROX-specific training** — zone 2 base building, pacing strategy, readiness score
- **Session history** — SQLite storage, pattern tracking over time
- **Web dashboard** — PWA that runs on iPhone, full-screen, no app store needed

## Architecture

```
Polar H10 (BLE) ──→ Python Server (Bleak) ──→ WebSocket ──→ iPhone Browser
                         │                                          │
                    ┌────┴────┐                              ┌──────┴──────┐
                    │  Coach  │                              │  Dashboard  │
                    │ Engine  │                              │  (PWA)      │
                    │  + TTS  │                              │             │
                    └─────────┘                              └─────────────┘
```

## Quick Start

1. **Run the server** (on your Windows machine):
   ```
   double-click start.bat
   ```
   Or:
   ```
   python scripts/server.py
   ```

2. **Open on iPhone**: Navigate to the IP shown (e.g., `http://192.168.1.x:8765`)

3. **Connect AirPods** to your iPhone

4. **Start session** → coach begins analyzing and speaking to you

## Data Flow

1. Polar H10 broadcasts HR + RR intervals via BLE (1Hz)
2. Python `ble_server.py` reads via Bleak, parses GATT data
3. `coach.py` analyzes: HR zones, HRV (RMSSD, frequency domain), trends, HYROX readiness
4. Coaching engine generates cues every 30s (configurable)
5. `edge-tts` converts cues to speech (MP3)
6. Audio streamed to browser via WebSocket → plays through AirPods
7. All data saved to SQLite for session history and pattern analysis

## Coaching Features

- **Zone guidance**: "You're in Z3, target is Z2. Slow down."
- **HYROX pacing**: "This is your race pace zone. Comfortable but purposeful."
- **HRV awareness**: "HRV is low. Focus on controlled breathing."
- **Recovery tracking**: Recovery score 0-100 based on HRV + HR trend
- **Training load**: TRIMP-based load calculation
- **HYROX readiness**: Aerobic base assessment for race day
- **Status updates**: Every 5 minutes with elapsed time, HR, zone

## Apple Watch Integration

Apple Watch Series 10 data can be incorporated via:
- **Health Export Pro** app → export HealthKit data → import for analysis
- **Real-time**: Watch broadcasts HR to iPhone HealthKit; future version can read this

## Rayban Meta Glasses

Meta Gen 2 glasses can receive audio coaching via Bluetooth audio from the iPhone. The TTS audio plays through whatever audio output the iPhone is connected to — AirPods or Meta glasses.

## Treadmill + Outdoor

Works identically on treadmill and outdoors. GPS data can be added via iPhone location services for outdoor pace tracking.

## Football Sessions

Wear H10 + Apple Watch during football. Session data is tagged and stored. Coach tracks:
- Time in each HR zone
- HRV recovery patterns
- Training load accumulation
- Weekly load trends

## Configuration

In the Settings tab:
- **Max HR**: Your actual max heart rate (default 190)
- **Resting HR**: Your resting heart rate (default 60)
- **Coach interval**: How often to give verbal cues (15s–2min)
- **Voice coaching**: Toggle on/off

## Tech Stack

- **Python 3.11**: Bleak (BLE), edge-tts (TTS), aiohttp (WebSocket server), numpy/scipy (HRV analysis), SQLite (storage)
- **Frontend**: Vanilla JS, Canvas charts, Web Bluetooth API (fallback), WebSocket, Web Audio API
- **No Node.js required** — pure Python backend

## Files

```
hyrox-coach-final/
├── scripts/
│   ├── server.py       # Main server (BLE + TTS + WebSocket + HTTP)
│   ├── ble_server.py   # Polar H10 BLE relay
│   └── coach.py        # Coaching engine + TTS
├── web/
│   ├── index.html      # PWA dashboard
│   └── manifest.json   # PWA manifest
├── sessions.db         # SQLite database (auto-created)
├── start.bat           # Windows startup script
└── README.md
```
