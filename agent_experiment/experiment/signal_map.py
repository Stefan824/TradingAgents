"""Map agent decisions (BUY / SELL / HOLD) to tri-state positions.

The Trading Agent outputs one of three strings via SignalProcessor:
  BUY, SELL, HOLD

The experiment uses tri-state positions aligned with deep-trading:
  1.0  = long  (BUY)
  0.0  = flat  (HOLD)
 -1.0  = short (SELL)
"""

from __future__ import annotations

import re

_DECISION_RE = re.compile(r"\b(BUY|SELL|HOLD)\b", re.IGNORECASE)


def parse_decision(raw: str) -> str:
    """Extract BUY/SELL/HOLD from potentially noisy agent output.

    Falls back to the raw stripped/uppercased string if the regex
    doesn't find a match, so callers can detect unrecognized outputs.
    """
    match = _DECISION_RE.search(raw)
    if match:
        return match.group(1).upper()
    return raw.strip().upper()


def decision_to_position(decision: str, hold_position: float = 0.0) -> float:
    """Convert a parsed decision to a tri-state position.

    Args:
        decision: One of "BUY", "SELL", "HOLD" (or unknown).
        hold_position: Value for HOLD (default 0.0 = flat).

    Returns:
        1.0, -1.0, or hold_position.
    """
    d = decision.strip().upper()
    if d == "BUY":
        return 1.0
    if d == "SELL":
        return -1.0
    return hold_position
