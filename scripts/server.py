"""
HYROX Coach — Main Server
Receives HR data from Web Bluetooth (browser) or BLE relay
Streams coaching + TTS back to dashboard via WebSocket
"""

import asyncio
import json
import os
import sys
import time
import signal
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coach import HYROXCoach, AthleteProfile, SessionState

import aiohttp
from aiohttp import web

DB_PATH = Path(__file__).parent.parent / "sessions.db"


def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL, ended_at TEXT, sport TEXT DEFAULT 'running',
            duration_seconds REAL, avg_hr REAL, max_hr INTEGER, min_hr INTEGER,
            rmssd REAL, training_load REAL, zone_times TEXT, hyrox_score INTEGER,
            notes TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hr_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
            timestamp REAL, hr INTEGER, rr_intervals TEXT, zone TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )""")
    conn.commit()
    conn.close()


def save_session(coach):
    conn = sqlite3.connect(str(DB_PATH))
    hr_list = list(coach.hr_history)
    if not hr_list: conn.close(); return
    analysis = coach.get_coaching_analysis()
    zone_times = json.dumps(coach.state.zone_time)
    avg = float(sum(hr_list)) / len(hr_list)
    cursor = conn.execute("""
        INSERT INTO sessions (started_at, ended_at, sport, duration_seconds, avg_hr, max_hr, min_hr, rmssd, training_load, zone_times, hyrox_score, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        datetime.now(timezone.utc).isoformat(),
        "running", coach.state.elapsed_seconds, avg, max(hr_list), min(hr_list),
        coach.state.rmssd, analysis.get("training_load", 0),
        zone_times, analysis.get("hyrox_readiness", {}).get("score", 0),
        json.dumps(analysis.get("suggestions", [])),
    ))
    session_id = cursor.lastrowid
    for i, hr in enumerate(hr_list):
        if i % 5 == 0:
            conn.execute("INSERT INTO hr_samples (session_id, timestamp, hr, zone) VALUES (?, ?, ?, ?)",
                (session_id, time.time() - (len(hr_list) - i), hr, coach._get_zone(hr)))
    conn.commit()
    conn.close()
    print(f"Session saved (ID: {session_id}, {len(hr_list)} samples)")


def get_session_history(limit=20):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@web.middleware
async def cors_middleware(request, handler):
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


class HYROXCoachApp:
    def __init__(self):
        self.coach = HYROXCoach()
        self.audio_queue = asyncio.Queue()
        self.ws_clients = set()
        self.session_start = None
        self.connected = False  # Will be set when first HR data arrives

    async def start(self):
        init_db()
        app = await self.create_web_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 8770, reuse_address=True)
        await site.start()

        local_ip = self._get_local_ip()
        print(f"\n{'='*60}")
        print(f"HYROX COACH — Server Running")
        print(f"{'='*60}")
        print(f"iPhone:     http://{local_ip}:8770")
        print(f"Local:      http://localhost:8770")
        print(f"{'='*60}")
        print(f"\nOpen on iPhone, connect Polar H10 via Web Bluetooth!\n")

        asyncio.create_task(self._coaching_loop())
        asyncio.create_task(self._audio_playback_loop())

        while True: await asyncio.sleep(1)

    def _get_local_ip(self):
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except: return "localhost"

    async def broadcast(self, data):
        if not self.ws_clients: return
        msg = json.dumps(data)
        dead = set()
        for ws in self.ws_clients:
            try: await ws.send_str(msg)
            except: dead.add(ws)
        self.ws_clients -= dead

    async def _coaching_loop(self):
        while True:
            await asyncio.sleep(5)
            if not self.connected or not self.coach.coach_enabled: continue
            if not self.coach.hr_history: continue

            cue = self.coach.generate_cue()
            if cue:
                print(f"Coach: {cue}")
                audio_b64 = await self.coach.speak(cue)
                if audio_b64:
                    await self.audio_queue.put({"type": "tts", "text": cue, "audio": audio_b64, "timestamp": datetime.now(timezone.utc).isoformat()})

            analysis = self.coach.get_coaching_analysis()
            await self.broadcast({"type": "coaching", "analysis": analysis})

    async def _audio_playback_loop(self):
        while True:
            item = await self.audio_queue.get()
            await self.broadcast(item)

    async def create_web_app(self):
        coach = self.coach
        app_ref = self

        async def websocket_handler(request):
            ws = web.WebSocketResponse(heartbeat=30)
            await ws.prepare(request)
            app_ref.ws_clients.add(ws)
            print(f"WebSocket connected from {request.remote} (total: {len(app_ref.ws_clients)})")
            try:
                await ws.send_str(json.dumps({
                    "type": "init", "connected": app_ref.connected,
                    "profile": {"max_hr": coach.profile.max_hr, "resting_hr": coach.profile.resting_hr},
                    "sessions": get_session_history(5),
                }))
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            await handle_ws(ws, data)
                        except json.JSONDecodeError: pass
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"WS error: {ws.exception()}")
            except Exception as e:
                print(f"WS handler error: {e}")
            finally:
                app_ref.ws_clients.discard(ws)
                print(f"WS disconnected (total: {len(app_ref.ws_clients)})")
            return ws

        async def handle_ws(ws, data):
            t = data.get("type")
            if t == "config":
                if "max_hr" in data: coach.profile.max_hr = int(data["max_hr"])
                if "resting_hr" in data: coach.profile.resting_hr = int(data["resting_hr"])
                if "coach_enabled" in data: coach.coach_enabled = bool(data["coach_enabled"])
                if "coach_interval" in data: coach.coach_interval_seconds = int(data["coach_interval"])
                await ws.send_str(json.dumps({"type": "config_ack", "profile": {"max_hr": coach.profile.max_hr, "resting_hr": coach.profile.resting_hr, "coach_enabled": coach.coach_enabled}}))
            elif t == "get_sessions":
                await ws.send_str(json.dumps({"type": "sessions", "sessions": get_session_history(20)}))
            elif t == "web_bluetooth_hr":
                hr = data.get("hr")
                rr = data.get("rr_intervals", [])
                if hr:
                    app_ref.connected = True
                    if app_ref.session_start is None: app_ref.session_start = time.time()
                    elapsed = time.time() - app_ref.session_start
                    coach.update(hr=hr, rr_intervals=rr, elapsed=elapsed)
                    # Build session stats
                    hr_list = list(coach.hr_history)
                    stats = {
                        "elapsed_seconds": round(elapsed),
                        "elapsed_formatted": f"{int(elapsed // 60)}:{int(elapsed % 60):02d}",
                        "current_hr": hr,
                        "avg_hr": round(float(sum(hr_list)) / len(hr_list)) if hr_list else hr,
                        "max_hr_session": max(hr_list) if hr_list else hr,
                        "min_hr_session": min(hr_list) if hr_list else hr,
                        "hr_zone": coach.state.zone,
                        "zone_distribution": dict(coach.state.zone_time),
                        "hrv": coach.calculate_hrv_metrics() if hasattr(coach, 'calculate_hrv_metrics') else {},
                        "total_samples": len(hr_list),
                    }
                    await app_ref.broadcast({
                        "type": "hr",
                        "data": {"hr": hr, "rr_intervals": rr, "timestamp": datetime.now(timezone.utc).isoformat()},
                        "stats": stats,
                        "coaching": coach.get_coaching_analysis(),
                    })
            elif t == "start_session":
                app_ref.session_start = time.time()
                coach.state = SessionState()
                coach.hr_history.clear()
                coach.rr_history.clear()
                app_ref.connected = False
                await app_ref.broadcast({"type": "session_started"})
            elif t == "stop_session":
                save_session(coach)
                app_ref.session_start = None
                app_ref.connected = False
                await app_ref.broadcast({"type": "session_stopped", "sessions": get_session_history(5)})

        async def health(request):
            return web.json_response({"status": "ok", "connected": app_ref.connected, "ws_clients": len(app_ref.ws_clients)})

        app = web.Application(middlewares=[cors_middleware])
        app.router.add_get("/ws", websocket_handler)
        app.router.add_get("/health", health)

        web_dir = Path(__file__).parent.parent / "web"
        if web_dir.exists():
            async def index(request):
                return web.FileResponse(web_dir / "index.html")
            async def static_files(request):
                fp = web_dir / request.match_info["path"]
                if fp.exists() and fp.is_file():
                    return web.FileResponse(fp)
                return web.FileResponse(web_dir / "index.html")
            app.router.add_get("/", index)
            app.router.add_get("/{path:.+}", static_files)

        return app


async def main():
    app = HYROXCoachApp()
    try:
        await app.start()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())
