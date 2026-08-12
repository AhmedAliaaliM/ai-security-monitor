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


ALL_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def is_within_auth_schedule(cfg: dict, now: datetime = None) -> bool:
    """
    Checks the SEPARATE 'auth_schedule' config (days + time window) that
    controls whether the identity/authorization pipeline runs at all.

    Motion detected WITHIN this schedule -> run the real face-recognition
    model, get a genuine authorized/unauthorized result.
    Motion detected OUTSIDE this schedule -> skip the model entirely,
    immediately treat the person as unauthorized (e.g. nobody should be
    there at 2am, so don't bother checking who it is).

    Default (if not configured): always active, every day - matches prior
    behavior so existing configs aren't silently changed.
    """
    now = now or datetime.now()
    auth_cfg = cfg.get("auth_schedule", {})

    allowed_days = auth_cfg.get("days", ALL_DAYS)
    today = now.strftime("%a")  # 'Mon', 'Tue', etc.
    if today not in allowed_days:
        return False

    start_s = auth_cfg.get("start", "00:00")
    end_s = auth_cfg.get("end", "23:59")
    start_t = _parse_hhmm(start_s)
    end_t = _parse_hhmm(end_s)
    return _in_window(now.time(), start_t, end_t)


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