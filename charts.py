"""
Chart capture and archiving -- window screenshots and chart library.

Extracted from market_bot_v26.py.
"""

from __future__ import annotations

import ctypes
import io
from datetime import time as dtime
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from bot_config import logger, CFG, now_et

# Guard win32gui imports -- only available on Windows
_WIN32_AVAILABLE = False
try:
    import win32gui, win32ui, win32con
    _WIN32_AVAILABLE = True
except ImportError:
    pass


def capture_window(title_keyword: str) -> Optional[Image.Image]:
    if not _WIN32_AVAILABLE:
        return None

    found = []

    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if CFG.TARGET_TICKER in title and title_keyword in title:
                found.append(hwnd)

    win32gui.EnumWindows(enum_handler, None)
    if not found:
        return None

    hwnd = found[0]
    try:
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        w, h = r - l, b - t
        if w <= 0 or h <= 0:
            return None

        hDC = win32gui.GetWindowDC(hwnd)
        mDC = win32ui.CreateDCFromHandle(hDC)
        sDC = mDC.CreateCompatibleDC()
        sBmp = win32ui.CreateBitmap()
        sBmp.CreateCompatibleBitmap(mDC, w, h)
        sDC.SelectObject(sBmp)
        ctypes.windll.user32.PrintWindow(hwnd, sDC.GetSafeHdc(), 2)
        bmpinfo = sBmp.GetInfo()
        bmpstr = sBmp.GetBitmapBits(True)
        img = Image.frombuffer("RGB", (bmpinfo["bmWidth"], bmpinfo["bmHeight"]), bmpstr, "raw", "BGRX", 0, 1)
        win32gui.DeleteObject(sBmp.GetHandle())
        sDC.DeleteDC()
        mDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hDC)
        return img
    except Exception as e:
        logger.warning(f"Window capture failed for '{title_keyword}': {e}")
        return None


def capture_triple_screen() -> Tuple[Optional[bytes], Optional[Image.Image]]:
    images = []
    for kw in [CFG.WINDOW_10M, CFG.WINDOW_1H, CFG.WINDOW_1D]:
        img = capture_window(kw)
        if img:
            images.append(img)

    if not images:
        return None, None

    total_w = sum(i.width for i in images)
    max_h = max(i.height for i in images)
    stitched = Image.new("RGB", (total_w, max_h))
    x = 0
    for im in images:
        stitched.paste(im, (x, 0))
        x += im.width

    stitched.thumbnail((1500, 1500))
    buf = io.BytesIO()
    stitched.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), stitched


class ChartLibrary:
    """
    Saves the 10-min chart screenshot once per hour during active trading.
    Builds a growing visual archive organized by date.
    """

    def __init__(self, base_dir: Path = Path("chart_library")):
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True)
        self.last_capture_hour: Optional[int] = None
        logger.info(f"ChartLibrary ready at {self.base_dir.resolve()} -- {self.get_total_days()} days archived")

    def should_capture(self) -> bool:
        """Returns True if we haven't captured in the current hour yet."""
        now = now_et()
        current_hour = now.hour

        if not (dtime(5, 30) <= now.time() <= dtime(17, 0)):
            return False

        if self.last_capture_hour == current_hour:
            return False

        return True

    def capture_and_save(self) -> bool:
        """Capture the 10-min chart and the full triple-screen, save both."""
        now = now_et()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M")

        day_dir = self.base_dir / date_str
        day_dir.mkdir(exist_ok=True)

        saved = False

        img_10m = capture_window(CFG.WINDOW_10M)
        if img_10m:
            path_10m = day_dir / f"ES_10m_{time_str}.jpg"
            img_10m.save(path_10m, format="JPEG", quality=90)
            logger.info(f"Chart saved: {path_10m.name}")
            saved = True

        img_bytes, img_obj = capture_triple_screen()
        if img_obj:
            path_triple = day_dir / f"ES_triple_{time_str}.jpg"
            img_obj.save(path_triple, format="JPEG", quality=90)
            logger.info(f"Triple-screen saved: {path_triple.name}")
            saved = True

        if saved:
            self.last_capture_hour = now.hour

        return saved

    def get_total_days(self) -> int:
        return sum(1 for d in self.base_dir.iterdir() if d.is_dir())

    def get_total_screenshots(self) -> int:
        count = 0
        for d in self.base_dir.iterdir():
            if d.is_dir():
                count += sum(1 for f in d.iterdir() if f.suffix == ".jpg")
        return count

    def get_day_screenshots(self, date_str: str) -> List[Path]:
        day_dir = self.base_dir / date_str
        if not day_dir.exists():
            return []
        return sorted(day_dir.glob("*.jpg"))

    def get_recent_days(self, n: int = 5) -> List[str]:
        dates = sorted(
            [d.name for d in self.base_dir.iterdir() if d.is_dir()],
            reverse=True,
        )
        return dates[:n]

    def get_status(self) -> str:
        days = self.get_total_days()
        total = self.get_total_screenshots()
        recent = self.get_recent_days(3)
        recent_str = ", ".join(recent) if recent else "None yet"
        return f"{days} days | {total} screenshots | Recent: {recent_str}"
