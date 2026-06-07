"""
HYROX Coach — Main Server v2
Integrates bleakheart BLE relay + multi-sport coaching + TTS + Web dashboard
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

from ble_relay import PolarH10Relay
from coach import LiveCoach, AthleteProfile, Sport

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
    hr_list = [h for _, h in coach.hr_history]
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
        coach.sport.value,
        coach.state.elapsed_seconds, avg, max(hr_list), min(hr_list),
        coach.state.rmssd, analysis.get("training_load", 0),
        zone_times, analysis.get("readiness", {}).get("score", 0),
        json.dumps(analysis.get("suggestions", [])),
    ))
    session_id = cursor.lastrowid
    for i, (ts, hr) in enumerate(coach.hr_history):
        if i % 5 == 0:
            conn.execute("INSERT INTO hr_samples (session_id, timestamp, hr, zone) VALUES (?, ?, ?, ?)",
                (session_id, ts, hr, coach.profile.zone_for(hr)))
    conn.commit()
    conn.close()
    print(f"Session saved (ID: {session_id})")


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
        self.coach = LiveCoach(sport=Sport.HYROX)
        self.audio_queue = asyncio.Queue()
        self.ws_clients = set()
        self.session_start = None
        self.ble_relay = None
        self.ble_task = None
        self.connected = False

    async def start(self):
        init_db()

        # Start BLE relay with callbacks
        self.ble_relay = PolarH10Relay(
            on_hr=self._on_hr,
            on_status=self._on_status,
        )
        self.ble_task = asyncio.create_task(self.ble_relay.run_forever())
        print("BLE relay started")

        # Create web app
        app = await self.create_web_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 8770, reuse_address=True)
        await site.start()

        local_ip = self._get_local_ip()
        print(f"\n{'='*60}")
        print(f"HYROX COACH v2 — Server Running")
        print(f"{'='*60}")
        print(f"iPhone:     http://{local_ip}:8770")
        print(f"Local:      http://localhost:8770")
        print(f"{'='*60}")

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

    # -- BLE callbacks --------------------------------------------------

    def _on_hr(self, hr, rr_intervals):
        """Called by ble_relay for every HR frame."""
        self.connected = True
        if self.session_start is None:
            self.session_start = time.time()
        self.coach.update(hr, rr_intervals)

    def _on_status(self, status):
        """Called by ble_relay for connection status updates."""
        state = status.get("state", "")
        print(f"BLE status: {status}")
        if state == "connected":
            self.broadcast_sync({"type": "device_info", "connected": True, "name": status.get("name")})
        elif state == "disconnected":
            self.connected = False
            self.broadcast_sync({"type": "device_info", "connected": False})
        elif state == "streaming":
            self.broadcast_sync({"type": "device_info", "streaming": True})

    def broadcast_sync(self, data):
        """Schedule a broadcast from a sync callback."""
        msg = json.dumps(data)
        for ws in list(self.ws_clients):
            asyncio.create_task(self._safe_send(ws, msg))

    async def _safe_send(self, ws, msg):
        try: await ws.send_str(msg)
        except: pass

    async def broadcast(self, data):
        if not self.ws_clients: return
        msg = json.dumps(data)
        dead = set()
        for ws in self.ws_clients:
            try: await ws.send_str(msg)
            except: dead.add(ws)
        self.ws_clients -= dead

    # -- Coaching loop --------------------------------------------------

    async def _coaching_loop(self):
        while True:
            await asyncio.sleep(5)
            if not self.connected: continue
            if not self.coach.hr_history: continue

            analysis = self.coach.get_coaching_analysis()
            await self.broadcast({"type": "coaching", "analysis": analysis})

            cue = self.coach.generate_cue()
            if cue:
                print(f"Coach: {cue}")
                audio_b64 = await self.coach.speak(cue)
                if audio_b64:
                    await self.audio_queue.put({"type": "tts", "text": cue, "audio": audio_b64, "timestamp": datetime.now(timezone.utc).isoformat()})

            # Also broadcast HR + stats for dashboard
            hr = self.coach.current_hr
            stats = self._build_stats()
            await self.broadcast({
                "type": "hr",
                "data": {"hr": hr, "timestamp": datetime.now(timezone.utc).isoformat()},
                "stats": stats,
            })

    async def _audio_playback_loop(self):
        while True:
            item = await self.audio_queue.get()
            await self.broadcast(item)

    def _build_stats(self):
        c = self.coach
        return {
            "avg_hr": round(c.avg_hr),
            "max_hr_session": c.max_hr_session,
            "hr_zone": c.zone,
            "hr_trend": c.hr_trend,
            "zone_distribution": {z: round(t) for z, t in c.zone_time.items()},
        }

    # -- Web app -------------------------------------------------------

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
                    "type": "init",
                    "connected": app_ref.connected,
                    "profile": {"max_hr": coach.profile.max_hr, "resting_hr": coach.profile.resting_hr},
                    "sport": coach.sport.value,
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
                if "sport" in data: coach.set_sport(data["sport"])
                if "max_hr" in data: coach.profile.max_hr = int(data["max_hr"])
                if "resting_hr" in data: coach.profile.resting_hr = int(data["resting_hr"])
                if "coach_enabled" in data: coach.coach_enabled = bool(data["coach_enabled"])
                if "coach_interval" in data: coach.coach_interval_seconds = int(data["coach_interval"])
                await ws.send_str(json.dumps({"type": "config_ack", "profile": {"max_hr": coach.profile.max_hr, "resting_hr": coach.profile.resting_hr, "coach_enabled": coach.coach_enabled, "sport": coach.sport.value}}))
            elif t == "get_sessions":
                await ws.send_str(json.dumps({"type": "sessions", "sessions": get_session_history(20)}))
            elif t == "web_bluetooth_hr":
                hr = data.get("hr")
                rr = data.get("rr_intervals", [])
                if hr:
                    app_ref.connected = True
                    if app_ref.session_start is None:
                        app_ref.session_start = time.time()
                    coach.update(hr, rr)
            elif t == "start_session":
                if "sport" in data: coach.set_sport(data["sport"])
                coach.reset()
                app_ref.session_start = time.time()
                app_ref.connected = False
                await app_ref.broadcast({"type": "session_started"})
            elif t == "stop_session":
                save_session(coach)
                app_ref.session_start = None
                app_ref.connected = False
                await app_ref.broadcast({"type": "session_stopped", "sessions": get_session_history(5)})

        async def health(request):
            return web.json_response({
                "status": "ok",
                "connected": app_ref.connected,
                "sport": coach.sport.value,
                "ws_clients": len(app_ref.ws_clients),
                "hr_samples": len(coach.hr_history),
            })

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
    finally:
        if app.ble_relay:
            app.ble_relay.stop()


if __name__ == "__main__":
    asyncio.run(main())
