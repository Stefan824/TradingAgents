"""Integration test: run the full trading pipeline with the mock LLM provider.

Verifies that the entire agent graph -- analysts, researchers, trader,
risk management, and portfolio manager -- completes end-to-end without
any real LLM or local model.

Usage:
    python test_local_integration.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


def run_test():
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "mock"
    config["deep_think_llm"] = "mock-deep"
    config["quick_think_llm"] = "mock-quick"
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1

    print("=" * 60)
    print("Integration Test: Full Pipeline with Mock LLM")
    print("=" * 60)
    print()

    selected_analysts = ["market", "social", "news", "fundamentals"]
    print(f"Analysts: {', '.join(selected_analysts)}")
    print(f"Provider: {config['llm_provider']}")
    print(f"Ticker: NVDA | Date: 2024-05-10")
    print()

    print("[1/3] Initializing TradingAgentsGraph...")
    ta = TradingAgentsGraph(
        selected_analysts=selected_analysts,
        debug=True,
        config=config,
    )
    print("      Graph initialized successfully.")
    print()

    print("[2/3] Running propagate('NVDA', '2024-05-10')...")
    print("-" * 60)
    final_state, decision = ta.propagate("NVDA", "2024-05-10")
    print("-" * 60)
    print("      Pipeline completed.")
    print()

    print("[3/3] Validating results...")
    errors = []

    # Validate reports
    report_fields = {
        "market_report": "Market Analyst",
        "sentiment_report": "Social Media Analyst",
        "news_report": "News Analyst",
        "fundamentals_report": "Fundamentals Analyst",
    }
    for field, label in report_fields.items():
        value = final_state.get(field, "")
        if value and len(value) > 10:
            print(f"  [OK] {label}: {len(value)} chars")
        else:
            errors.append(f"{label} report is empty or too short")
            print(f"  [FAIL] {label}: empty or missing")

    # Validate investment debate
    debate = final_state.get("investment_debate_state", {})
    if debate.get("bull_history", "").strip():
        print(f"  [OK] Bull Researcher: contributed to debate")
    else:
        errors.append("Bull Researcher history is empty")
        print(f"  [FAIL] Bull Researcher: no contribution")

    if debate.get("bear_history", "").strip():
        print(f"  [OK] Bear Researcher: contributed to debate")
    else:
        errors.append("Bear Researcher history is empty")
        print(f"  [FAIL] Bear Researcher: no contribution")

    if debate.get("judge_decision", "").strip():
        print(f"  [OK] Research Manager: made decision")
    else:
        errors.append("Research Manager judge_decision is empty")
        print(f"  [FAIL] Research Manager: no decision")

    # Validate investment plan
    if final_state.get("investment_plan", "").strip():
        print(f"  [OK] Investment Plan: present")
    else:
        errors.append("Investment plan is empty")
        print(f"  [FAIL] Investment Plan: missing")

    # Validate trader
    if final_state.get("trader_investment_plan", "").strip():
        print(f"  [OK] Trader: made investment plan")
    else:
        errors.append("Trader investment plan is empty")
        print(f"  [FAIL] Trader: no plan")

    # Validate risk debate
    risk = final_state.get("risk_debate_state", {})
    for role in ("aggressive_history", "conservative_history", "neutral_history"):
        label = role.replace("_history", "").capitalize()
        if risk.get(role, "").strip():
            print(f"  [OK] {label} Risk Analyst: contributed")
        else:
            errors.append(f"{label} risk history is empty")
            print(f"  [FAIL] {label} Risk Analyst: no contribution")

    if risk.get("judge_decision", "").strip():
        print(f"  [OK] Risk Judge (Portfolio Manager): made decision")
    else:
        errors.append("Risk judge decision is empty")
        print(f"  [FAIL] Risk Judge: no decision")

    # Validate final decision
    if final_state.get("final_trade_decision", "").strip():
        print(f"  [OK] Final Trade Decision: present")
    else:
        errors.append("Final trade decision is empty")
        print(f"  [FAIL] Final Trade Decision: missing")

    # Validate signal extraction
    decision_upper = decision.strip().upper() if decision else ""
    if decision_upper in ("BUY", "SELL", "HOLD"):
        print(f"  [OK] Signal Extraction: {decision_upper}")
    else:
        errors.append(f"Signal extraction returned unexpected: '{decision}'")
        print(f"  [FAIL] Signal Extraction: got '{decision}', expected BUY/SELL/HOLD")

    print()
    print("=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        print("=" * 60)
        return 1
    else:
        print("PASSED: All checks passed")
        print(f"  Decision: {decision_upper}")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(run_test())
