"""Mock LLM client for integration testing without a real model."""

from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .base_client import BaseLLMClient


MOCK_MARKET_REPORT = """## Market Analysis Report

**Trend Overview**: The stock shows a moderate uptrend over the past 30 days with increasing volume. The 50-day SMA is above the 200-day SMA, confirming a bullish golden cross pattern.

**Key Technical Indicators**:
- RSI: 62 (neutral-bullish, not overbought)
- MACD: Positive crossover observed, histogram expanding
- Bollinger Bands: Price trading in upper half, bandwidth expanding
- ATR: Volatility increasing slightly, suggesting momentum building

**Volume Analysis**: Volume has been 15% above the 20-day average, supporting the current uptrend.

| Indicator | Value | Signal |
|-----------|-------|--------|
| RSI (14) | 62.3 | Neutral-Bullish |
| MACD | +1.24 | Bullish Crossover |
| 50 SMA | Above 200 SMA | Golden Cross |
| ATR (14) | 2.8% | Moderate Volatility |

FINAL TRANSACTION PROPOSAL: **BUY**"""

MOCK_SENTIMENT_REPORT = """## Social Sentiment Analysis

**Overall Sentiment**: Moderately positive across major social platforms.

**Key Findings**:
- Twitter/X mentions up 23% week-over-week with 68% positive sentiment
- Reddit discussion volume elevated; bullish posts outnumber bearish 3:1
- Institutional investor commentary skews cautiously optimistic
- No significant negative viral content detected

| Platform | Sentiment | Volume Change |
|----------|-----------|---------------|
| Twitter/X | 68% Positive | +23% WoW |
| Reddit | 75% Positive | +18% WoW |
| StockTwits | 61% Positive | +12% WoW |

FINAL TRANSACTION PROPOSAL: **BUY**"""

MOCK_NEWS_REPORT = """## News Analysis Report

**Key Developments**:
1. Company reported Q1 earnings beating consensus estimates by 8%
2. New product launch received positive industry coverage
3. Sector-wide tailwinds from favorable regulatory developments
4. No significant insider selling detected in recent filings

**Macro Environment**: Stable interest rate outlook supports growth-oriented investments. Recent economic data indicates resilient consumer spending.

| Factor | Impact | Confidence |
|--------|--------|------------|
| Earnings Beat | Positive | High |
| Product Launch | Positive | Medium |
| Regulatory | Positive | Medium |
| Macro Outlook | Neutral-Positive | Medium |

FINAL TRANSACTION PROPOSAL: **BUY**"""

MOCK_FUNDAMENTALS_REPORT = """## Fundamentals Analysis

**Financial Health**: Strong balance sheet with healthy cash reserves and manageable debt levels.

**Key Metrics**:
- Revenue growth: 12% YoY
- Gross margin: 65% (expanding)
- Operating margin: 28% (stable)
- Free cash flow positive and growing
- Debt-to-equity: 0.45 (conservative)
- Current ratio: 2.1 (strong liquidity)

**Valuation**: Trading at a reasonable P/E relative to growth rate (PEG ~1.2).

| Metric | Value | Trend |
|--------|-------|-------|
| Revenue Growth | 12% YoY | Improving |
| Gross Margin | 65% | Expanding |
| FCF Yield | 3.8% | Stable |
| P/E Ratio | 28x | Fair |
| Debt/Equity | 0.45 | Conservative |

FINAL TRANSACTION PROPOSAL: **BUY**"""

MOCK_BULL_ARGUMENT = """The investment case here is compelling. Revenue growth of 12% YoY with expanding margins demonstrates operational leverage. The golden cross on the technical chart aligns with improving fundamentals, and positive social sentiment suggests the market is beginning to recognize the value proposition. The recent earnings beat provides a catalyst, and the PEG ratio of 1.2 suggests the stock is reasonably priced relative to its growth trajectory. I strongly advocate for a BUY position."""

MOCK_BEAR_ARGUMENT = """While the fundamentals appear solid on the surface, the RSI at 62 suggests we are approaching overbought territory. The elevated social media activity could indicate speculative froth rather than fundamental conviction. Additionally, the sector tailwinds may already be priced in. However, I acknowledge the earnings beat is a genuine positive signal. My concern is primarily around timing and valuation stretch rather than fundamental deterioration."""

MOCK_RESEARCH_MANAGER_DECISION = """After carefully evaluating both perspectives, I side with the bull analyst. The convergence of strong earnings, positive technical momentum, and improving fundamentals creates a favorable risk-reward profile.

**Recommendation: BUY**

**Strategic Actions**:
1. Initiate a position at current levels
2. Set a stop-loss at 5% below entry
3. Target a 15% upside over the next quarter
4. Monitor upcoming earnings and sector developments for position adjustment

FINAL TRANSACTION PROPOSAL: **BUY**"""

MOCK_TRADER_DECISION = """Based on the comprehensive analysis from the research team, I am executing a BUY recommendation.

**Trade Plan**:
- Action: BUY
- Position Size: Moderate (not full allocation due to elevated RSI)
- Entry: At current market price
- Stop-Loss: 5% below entry
- Take-Profit Target: 15% above entry
- Time Horizon: 1-3 months

The research team's bullish consensus, supported by strong earnings, positive technicals, and favorable sentiment, provides sufficient conviction for this trade.

FINAL TRANSACTION PROPOSAL: **BUY**"""

MOCK_AGGRESSIVE_RISK = """The upside potential here significantly outweighs the downside risk. With 12% revenue growth, expanding margins, and a technical golden cross, we should be taking a larger position than suggested. The 5% stop-loss is too tight for the volatility profile; I would advocate for an 8% stop to avoid being shaken out. The conservative concern about RSI is overblown -- in strong uptrends, RSI can remain elevated for extended periods. We should be aggressive here."""

MOCK_CONSERVATIVE_RISK = """While I acknowledge the positive signals, prudent risk management demands caution. The position size should be limited, and the stop-loss should remain tight. The market has been running hot, and mean reversion is always a risk. I recommend a smaller initial position with plans to add on any pullback, rather than going all-in at these levels."""

MOCK_NEUTRAL_RISK = """Both the aggressive and conservative viewpoints have merit. A balanced approach would be to initiate a moderate position as the trader suggests, with the 5% stop-loss as a reasonable compromise. The risk-reward is favorable but not extraordinary. I support the BUY recommendation with the proposed position sizing."""

MOCK_RISK_JUDGE_DECISION = """After evaluating all three risk perspectives, I recommend proceeding with the BUY.

**Final Risk-Adjusted Plan**:
- Action: BUY
- Position Size: Moderate allocation
- Stop-Loss: 6% below entry (compromise between aggressive and conservative)
- Take-Profit: 15% target with trailing stop
- Risk Rating: Moderate

The aggressive analyst makes a valid point about the strong trend, while the conservative analyst's caution about position sizing is prudent. The neutral analyst's balanced view aligns closest with optimal risk management. Proceeding with a moderate position at current levels.

FINAL TRANSACTION PROPOSAL: **BUY**"""


def _extract_text(messages: List[BaseMessage]) -> str:
    parts = []
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
    return " ".join(parts)


def _route_response(text: str) -> str:
    t = text.lower()

    if "extract the investment decision" in t:
        return "BUY"

    if "analyzing financial markets" in t or "technical indicator" in t:
        return MOCK_MARKET_REPORT
    if "social media" in t and "sentiment" in t and "analyst" not in t:
        return MOCK_SENTIMENT_REPORT
    if "news" in t and "researcher" in t and "world affairs" in t:
        return MOCK_NEWS_REPORT
    if "fundamental information" in t or "financial documents" in t:
        return MOCK_FUNDAMENTALS_REPORT

    if "bull analyst" in t:
        return MOCK_BULL_ARGUMENT
    if "bear analyst" in t:
        return MOCK_BEAR_ARGUMENT

    if "portfolio manager and debate facilitator" in t:
        return MOCK_RESEARCH_MANAGER_DECISION
    if "risk management judge" in t:
        return MOCK_RISK_JUDGE_DECISION

    if "trading agent" in t and "investment decision" in t:
        return MOCK_TRADER_DECISION

    if "aggressive risk analyst" in t:
        return MOCK_AGGRESSIVE_RISK
    if "conservative risk analyst" in t:
        return MOCK_CONSERVATIVE_RISK
    if "neutral risk analyst" in t:
        return MOCK_NEUTRAL_RISK

    if "expert financial analyst" in t and "reviewing trading" in t:
        return "Reflection: The analysis was thorough and well-supported by data."

    return "Analysis complete. FINAL TRANSACTION PROPOSAL: **BUY**"


class MockChatModel(BaseChatModel):
    """A mock chat model that returns canned trading analysis responses.

    Detects agent role from prompt content and returns appropriate responses.
    Supports bind_tools() by always returning text (no tool_calls), causing
    analyst nodes to skip tool execution and proceed directly to report output.
    """

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_text = _extract_text(messages)
        content = _route_response(prompt_text)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for integration testing."""

    def get_llm(self) -> Any:
        return MockChatModel()

    def validate_model(self) -> bool:
        return True
