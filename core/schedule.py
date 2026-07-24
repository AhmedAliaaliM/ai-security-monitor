"""
Schedule utility: decides whether the system should currently run in
'ai_mode' (full pipeline: router + hazard/identity detection + alerts)
or 'cctv_mode' (plain motion-triggered recording, no AI, no alerts).

The camera keeps capturing/monitoring for motion in BOTH modes — only the
processing behavior downstream changes.
"""

from datetime import datetime, time as dtime


def _parse_hhmm(s: str) -> dtime:
    h, m = map(int, s.split(":"))
    return dtime(hour=h, minute=m)


def _in_window(now: dtime, start: dtime, end: dtime) -> bool:
    if start <= end:
        return start <= now <= end
    # window wraps past midnight, e.g. 22:00 -> 06:00
    return now >= start or now <= end


def get_current_mode(cfg: dict, now: datetime = None) -> str:
    """Returns 'ai_mode' or 'cctv_mode' based on the camera's schedule config."""
    now = now or datetime.now()
    now_t = now.time()

    for start_s, end_s in cfg.get("schedule", {}).get("ai_mode", []):
        start_t = _parse_hhmm(start_s)
        end_t = _parse_hhmm(end_s)
        if _in_window(now_t, start_t, end_t):
            return "ai_mode"

    return "cctv_mode"
