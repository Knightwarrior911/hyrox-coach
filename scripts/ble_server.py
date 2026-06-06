"""
Polar H10 BLE Relay Server
Reads HR, RR intervals, ECG from Polar H10 via Bleak
Streams data over WebSocket to the web dashboard
"""

import asyncio
import json
import time
import signal
import sys
from datetime import datetime, timezone
from collections import deque
from typing import Optional

import numpy as np
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic

# ---- Polar H10 GATT UUIDs ----
HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
ECG_SERVICE_UUID = "fb005c80-02e7-f387-1cad-8acd2d8df0c8"
ECG_WRITE_CHAR_UUID = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"
ECG_DATA_CHAR_UUID = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
BATTERY_SERVICE_UUID = "0000180f-0000-1000-8000-00805f9b34fb"
BATTERY_LEVEL_CHAR_UUID = "00002a19-0000-1000-8000-00805f9b34fb"

# ECG stream commands
ECG_START_CMD = bytes([0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
ECG_STOP_CMD = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])


class PolarH10Relay:
    def __init__(self):
        self.client: Optional[BleakClient] = None
        self.device_address: Optional[str] = None
        self.connected = False
        self.ws_clients: set = set()
        self.session_start: Optional[float] = None

        self.rr_intervals: deque = deque(maxlen=300)
        self.hr_history: deque = deque(maxlen=3600)
        self.ecg_buffer: deque = deque(maxlen=1300)
        self.rmssd_history: deque = deque(maxlen=300)
        self.zone_time: dict = {"Z1": 0, "Z2": 0, "Z3": 0, "Z4": 0, "Z5": 0}
        self.last_zone_update: Optional[float] = None

        self.max_hr: int = 190
        self.resting_hr: int = 60

    def hr_zone(self, hr: int) -> str:
        pct = hr / self.max_hr
        if pct < 0.6: return "Z1"
        elif pct < 0.7: return "Z2"
        elif pct < 0.8: return "Z3"
        elif pct < 0.9: return "Z4"
        else: return "Z5"

    def calculate_rmssd(self) -> Optional[float]:
        if len(self.rr_intervals) < 3: return None
        rr = np.array(list(self.rr_intervals))
        successive_diffs = np.diff(rr)
        return float(np.sqrt(np.mean(successive_diffs ** 2)))

    def calculate_hrv_metrics(self) -> dict:
        if len(self.rr_intervals) < 5: return {}
        rr = np.array(list(self.rr_intervals))
        successive_diffs = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
        sdnn = float(np.std(rr))
        p50 = float(np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100)

        if len(rr) >= 64:
            rr_interp = np.interp(np.linspace(0, len(rr) - 1, 256), np.arange(len(rr)), rr - np.mean(rr))
            fft = np.abs(np.fft.rfft(rr_interp)) ** 2
            freqs = np.fft.rfftfreq(256, d=1.0)
            vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.4)
            vlf_power = float(np.sum(fft[vlf_mask])) if np.any(vlf_mask) else 0
            lf_power = float(np.sum(fft[lf_mask])) if np.any(lf_mask) else 0
            hf_power = float(np.sum(fft[hf_mask])) if np.any(hf_mask) else 0
            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else None
        else:
            vlf_power = lf_power = hf_power = lf_hf_ratio = None

        return {
            "rmssd": round(rmssd, 1), "sdnn": round(sdnn, 1), "p50": round(p50, 1),
            "vlf_power": round(vlf_power, 1) if vlf_power else None,
            "lf_power": round(lf_power, 1) if lf_power else None,
            "hf_power": round(hf_power, 1) if hf_power else None,
            "lf_hf_ratio": round(lf_hf_ratio, 2) if lf_hf_ratio else None,
        }

    def get_session_stats(self) -> dict:
        elapsed = time.time() - self.session_start if self.session_start else 0
        hr_list = [h["hr"] for h in self.hr_history]
        return {
            "elapsed_seconds": round(elapsed),
            "elapsed_formatted": f"{int(elapsed // 60)}:{int(elapsed % 60):02d}",
            "current_hr": hr_list[-1] if hr_list else None,
            "avg_hr": round(np.mean(hr_list)) if hr_list else None,
            "max_hr_session": max(hr_list) if hr_list else None,
            "min_hr_session": min(hr_list) if hr_list else None,
            "hr_zone": self.hr_zone(hr_list[-1]) if hr_list else None,
            "zone_distribution": dict(self.zone_time),
            "hrv": self.calculate_hrv_metrics(),
            "total_samples": len(self.hr_history),
        }

    async def broadcast(self, data: dict):
        if not self.ws_clients: return
        msg = json.dumps(data)
        disconnected = set()
        for ws in self.ws_clients:
            try: await ws.send_str(msg)
            except Exception: disconnected.add(ws)
        self.ws_clients -= disconnected

    def parse_hr_data(self, data: bytearray) -> dict:
        flags = data[0]
        hr_format_16bit = flags & 0x01
        sensor_contact = (flags >> 1) & 0x03
        energy_expended = (flags >> 3) & 0x01
        rr_present = (flags >> 4) & 0x01
        result = {"sensor_contact": sensor_contact == 2, "timestamp": datetime.now(timezone.utc).isoformat()}
        offset = 1
        if hr_format_16bit:
            hr = int.from_bytes(data[offset:offset + 2], byteorder="little")
            offset += 2
        else:
            hr = data[offset]
            offset += 1
        result["hr"] = hr
        if energy_expended:
            result["energy_expended"] = int.from_bytes(data[offset:offset + 2], byteorder="little")
            offset += 2
        rr_intervals = []
        if rr_present:
            while offset + 1 < len(data):
                rr_raw = int.from_bytes(data[offset:offset + 2], byteorder="little")
                rr_intervals.append(round(rr_raw * 1000 / 1024, 1))
                offset += 2
            result["rr_intervals"] = rr_intervals
            for rr in rr_intervals: self.rr_intervals.append(rr)
        return result

    def parse_ecg_data(self, data: bytearray) -> list:
        samples = []
        if len(data) < 10: return samples
        offset = 8
        while offset + 2 < len(data):
            raw = data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16)
            if raw >= 0x800000: raw -= 0x1000000
            samples.append(round(raw * 32000 / 16384, 1))
            offset += 3
        return samples

    async def hr_notification_handler(self, sender: BleakGATTCharacteristic, data: bytearray):
        parsed = self.parse_hr_data(data)
        if parsed.get("hr") and self.last_zone_update:
            zone = self.hr_zone(parsed["hr"])
            dt = time.time() - self.last_zone_update
            self.zone_time[zone] = self.zone_time.get(zone, 0) + dt
        self.last_zone_update = time.time()
        self.hr_history.append({"hr": parsed["hr"], "ts": time.time()})
        await self.broadcast({"type": "hr", "data": parsed, "stats": self.get_session_stats()})

    async def ecg_notification_handler(self, sender: BleakGATTCharacteristic, data: bytearray):
        samples = self.parse_ecg_data(data)
        if samples:
            self.ecg_buffer.extend(samples)
            await self.broadcast({"type": "ecg", "samples": samples, "buffer_size": len(self.ecg_buffer)})

    async def scan_for_polar(self, timeout: float = 10.0) -> Optional[str]:
        print(f"Scanning for Polar devices (timeout: {timeout}s)...")
        devices = await BleakScanner.discover(timeout=timeout)
        for device in devices:
            name = device.name or ""
            if "polar" in name.lower() or "h10" in name.lower():
                print(f"Found: {name} ({device.address})")
                return device.address
        devices = await BleakScanner.discover(timeout=timeout, service_uuids=[HR_SERVICE_UUID])
        for device in devices:
            if device.name:
                print(f"Found by HR service: {device.name} ({device.address})")
                return device.address
        print("No Polar H10 found")
        return None

    async def connect(self, address: Optional[str] = None):
        if not address:
            address = await self.scan_for_polar()
            if not address: raise Exception("No Polar H10 found")
        self.device_address = address
        print(f"Connecting to {address}...")
        self.client = BleakClient(address, disconnected_callback=self._on_disconnect)
        await self.client.connect()
        self.connected = True
        self.session_start = time.time()
        self.last_zone_update = time.time()
        print(f"Connected to Polar H10 ({address})")
        try:
            battery_data = await self.client.read_gatt_char(BATTERY_LEVEL_CHAR_UUID)
            battery_level = battery_data[0]
            print(f"Battery: {battery_level}%")
            await self.broadcast({"type": "device_info", "battery": battery_level, "address": address, "connected": True})
        except Exception as e:
            print(f"Could not read battery: {e}")

    def _on_disconnect(self, client):
        print("Device disconnected unexpectedly")
        self.connected = False
        asyncio.create_task(self.broadcast({"type": "device_info", "connected": False, "message": "Device disconnected"}))

    async def start_hr_stream(self):
        if not self.connected: raise Exception("Not connected")
        print("Starting heart rate stream...")
        # Discover the actual HR measurement characteristic
        hr_char = None
        for service in self.client.services:
            if service.uuid.lower() == HR_SERVICE_UUID.lower():
                for char in service.characteristics:
                    if "notify" in char.properties:
                        hr_char = char
                        print(f"  Found HR notify char: {char.uuid}")
                        break
            if hr_char:
                break
        if not hr_char:
            # Fallback: try the standard UUID
            hr_char = HR_MEASUREMENT_CHAR_UUID
            print(f"  Using standard HR char UUID: {hr_char}")
        await self.client.start_notify(hr_char, self.hr_notification_handler)
        print("HR stream active")

    async def start_ecg_stream(self):
        if not self.connected: raise Exception("Not connected")
        print("Starting ECG stream...")
        await self.client.write_gatt_char(ECG_WRITE_CHAR_UUID, ECG_START_CMD)
        await self.client.start_notify(ECG_DATA_CHAR_UUID, self.ecg_notification_handler)
        print("ECG stream active")

    async def stop_ecg_stream(self):
        if self.connected:
            try:
                await self.client.write_gatt_char(ECG_WRITE_CHAR_UUID, ECG_STOP_CMD)
                await self.client.stop_notify(ECG_DATA_CHAR_UUID)
            except Exception: pass

    async def disconnect(self):
        if self.connected and self.client:
            try:
                await self.stop_ecg_stream()
                await self.client.stop_notify(HR_MEASUREMENT_CHAR_UUID)
                await self.client.disconnect()
            except Exception: pass
        self.connected = False
        print("Disconnected")


def get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"
