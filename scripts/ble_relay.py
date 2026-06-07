"""
ble_relay.py - Polar H10 BLE relay built on the `bleakheart` library.

    Polar H10  ->  bleakheart (HeartRate / BatteryLevel)  ->  on_hr / on_status

Drop-in replacement for the raw-Bleak relay: it exposes the same
PolarH10Relay(on_hr, on_status) interface used by server.py.

Install:   pip install bleakheart
Reference: https://github.com/fsmeraldi/bleakheart
"""

import asyncio
from typing import Callable, List, Optional

from bleak import BleakScanner, BleakClient
from bleakheart import HeartRate, BatteryLevel

HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"


class PolarH10Relay:
    """Connects to a Polar H10 (or any standard BLE HR strap) and streams
    heart rate + RR intervals to callbacks.

    Args:
        on_hr:     called for every HR frame as on_hr(hr_bpm, rr_intervals_ms),
                   where rr_intervals_ms is a (possibly empty) list in ms.
        on_status: optional, called with a dict describing connection state,
                   battery level and skin-contact changes.
        address:   optional BLE address to skip scanning.
        name_hint: case-insensitive substring matched against the device name.
    """

    def __init__(
        self,
        on_hr: Callable[[int, List[int]], None],
        on_status: Optional[Callable[[dict], None]] = None,
        address: Optional[str] = None,
        name_hint: str = "polar",
        reconnect_delay: float = 5.0,
    ):
        self.on_hr = on_hr
        self.on_status = on_status
        self.address = address
        self.name_hint = name_hint.lower()
        self.reconnect_delay = reconnect_delay
        self._stop = asyncio.Event()
        self._disconnected = asyncio.Event()

    # -- helpers --------------------------------------------------------

    def _status(self, **kwargs):
        if self.on_status:
            try:
                self.on_status(kwargs)
            except Exception as e:
                print(f"[ble_relay] on_status error: {e}")

    async def _find_device(self):
        """Find the strap by address, then by name hint, then by HR service."""
        if self.address:
            self._status(state="scanning", detail=f"address {self.address}")
            return await BleakScanner.find_device_by_address(self.address, timeout=10.0)

        self._status(state="scanning", detail="by name")
        device = await BleakScanner.find_device_by_filter(
            lambda d, adv: bool(d.name) and self.name_hint in d.name.lower(),
            timeout=10.0,
        )
        if device is None:
            self._status(state="scanning", detail="by HR service")
            device = await BleakScanner.find_device_by_filter(
                lambda d, adv: HR_SERVICE_UUID
                in [u.lower() for u in (adv.service_uuids or [])],
                timeout=10.0,
            )
        return device

    # -- bleakheart callbacks -------------------------------------------

    def _hr_frame(self, frame):
        # frame == ('HR', tstamp_ns, (avg_hr, [rr_ms, ...]), energy)
        try:
            _, _, (hr, rr_list), _ = frame
        except (ValueError, TypeError):
            return
        if hr:
            self.on_hr(int(hr), list(rr_list or []))

    def _on_good_contact(self):
        self._status(contact=True)

    def _on_lost_contact(self):
        self._status(contact=False)

    # -- connection lifecycle -------------------------------------------

    async def _session(self):
        device = await self._find_device()
        if device is None:
            self._status(state="not_found")
            return

        self._disconnected.clear()

        def disconnected_callback(_client):
            self._disconnected.set()

        async with BleakClient(device, disconnected_callback=disconnected_callback) as client:
            self._status(state="connected", name=getattr(device, "name", None))

            try:
                battery = await BatteryLevel(client).read()
                self._status(battery=battery)
            except Exception:
                pass  # not all straps expose battery

            heartrate = HeartRate(
                client,
                callback=self._hr_frame,        # unpack=False -> full RR list per frame
                contact_callback=self._on_good_contact,
                contact_lost_callback=self._on_lost_contact,
                instant_rate=False,
                unpack=False,
            )
            await heartrate.start_notify()
            self._status(state="streaming")

            stop_task = asyncio.create_task(self._stop.wait())
            drop_task = asyncio.create_task(self._disconnected.wait())
            await asyncio.wait({stop_task, drop_task}, return_when=asyncio.FIRST_COMPLETED)
            for t in (stop_task, drop_task):
                t.cancel()

            if client.is_connected:
                try:
                    await heartrate.stop_notify()
                except Exception:
                    pass

        self._status(state="disconnected")

    async def run_forever(self):
        """Connect and keep the strap connected, auto-reconnecting on drops."""
        while not self._stop.is_set():
            try:
                await self._session()
            except Exception as e:
                self._status(state="error", detail=str(e))
            if self._stop.is_set():
                break
            self._status(state="reconnecting", detail=f"{self.reconnect_delay:.0f}s")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.reconnect_delay)
            except asyncio.TimeoutError:
                pass

    def stop(self):
        self._stop.set()
