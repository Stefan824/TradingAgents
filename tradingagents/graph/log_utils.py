"""Utilities for logging and exporting trading agent state."""

import json
from pathlib import Path


def _state_to_md(state: dict) -> list[str]:
    """Convert a single date's state dict to markdown lines."""
    parts = []
    parts.append(
        f"**Company:** {state.get('company_of_interest', 'N/A')} | **Trade Date:** {state.get('trade_date', 'N/A')}\n"
    )

    for key in ["market_report", "sentiment_report", "news_report", "fundamentals_report"]:
        if key in state and state[key]:
            title = key.replace("_", " ").title()
            parts.append(f"\n---\n## {title}\n\n{state[key]}\n")

    if "investment_debate_state" in state:
        ids = state["investment_debate_state"]
        parts.append("\n---\n## Investment Debate\n\n")
        for subkey in [
            "bull_history",
            "bear_history",
            "current_response",
            "judge_decision",
            "trader_investment_decision",
        ]:
            if subkey in ids and ids[subkey]:
                title = subkey.replace("_", " ").title()
                parts.append(f"### {title}\n\n{ids[subkey]}\n\n")

    if "risk_debate_state" in state:
        rds = state["risk_debate_state"]
        parts.append("\n---\n## Risk Debate\n\n")
        for subkey in [
            "aggressive_history",
            "conservative_history",
            "neutral_history",
            "judge_decision",
        ]:
            if subkey in rds and rds[subkey]:
                title = subkey.replace("_", " ").title()
                parts.append(f"### {title}\n\n{rds[subkey]}\n\n")

    for key in ["investment_plan", "final_trade_decision"]:
        if key in state and state[key]:
            title = key.replace("_", " ").title()
            parts.append(f"\n---\n## {title}\n\n{state[key]}\n")

    return parts


def full_states_json_to_md(
    json_path: str | Path, md_path: str | Path | None = None
) -> Path:
    """
    Convert a full_states_log JSON file to Markdown.

    Args:
        json_path: Path to the JSON file.
        md_path: Optional output path. Defaults to same dir, same stem with .md.

    Returns:
        Path to the written MD file.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    if md_path is None:
        md_path = json_path.with_suffix(".md")
    else:
        md_path = Path(md_path)

    md_path.parent.mkdir(parents=True, exist_ok=True)

    md_parts = []
    for date_key, state in data.items():
        if not isinstance(state, dict):
            continue
        md_parts.append(f"# Full State Log — {date_key}\n")
        md_parts.extend(_state_to_md(state))
        md_parts.append("\n")

    with open(md_path, "w") as f:
        f.write("\n".join(md_parts))

    return md_path
