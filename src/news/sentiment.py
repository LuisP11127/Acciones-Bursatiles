"""Sentimiento de titulares: VADER con vocabulario financiero (o solo el léxico si
VADER no está instalado). Es un análisis léxico: entiende frases simples, no
ironía ni contexto complejo."""

from __future__ import annotations

import math
import re

FINANCIAL_LEXICON = {
    "beat": 1.6, "beats": 1.8, "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0, "outperform": 1.8,
    "overweight": 1.4, "surge": 2.2, "surges": 2.2, "soar": 2.5, "soars": 2.5, "jump": 1.6, "jumps": 1.8,
    "rally": 1.8, "rallies": 1.8, "record": 1.4, "bullish": 2.4, "buyback": 1.6, "raises": 1.2,
    "raised": 1.0, "growth": 1.4, "profit": 1.4, "profitable": 1.8, "expands": 1.2, "expansion": 1.2,
    "partnership": 1.2, "approval": 1.8, "approved": 1.8, "breakthrough": 2.0, "gains": 1.4, "tops": 1.4,
    "strong": 1.4, "dividend": 0.8, "wins": 1.6,
    "miss": -1.8, "misses": -1.8, "missed": -1.8, "downgrade": -2.0, "downgrades": -2.0,
    "downgraded": -2.0, "underperform": -1.8, "underweight": -1.4, "plunge": -2.6, "plunges": -2.6,
    "tumble": -2.2, "tumbles": -2.2, "slump": -2.0, "slumps": -2.0, "sinks": -2.0, "falls": -1.2,
    "drops": -1.2, "bearish": -2.4, "lawsuit": -2.0, "sued": -2.0, "probe": -1.6, "investigation": -1.6,
    "recall": -1.8, "layoffs": -1.6, "cuts": -1.2, "bankruptcy": -3.4, "fraud": -3.0, "warning": -1.4,
    "weak": -1.6, "loss": -1.5, "losses": -1.5, "decline": -1.3, "declines": -1.3, "selloff": -2.0,
    "sell-off": -2.0, "halt": -1.6, "fined": -1.8, "delay": -1.0, "delays": -1.0, "antitrust": -1.4,
    "subpoena": -1.8,
}


class SentimentAnalyzer:
    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()
            self._vader.lexicon.update(FINANCIAL_LEXICON)
            self.method = "vader+lexico_financiero"
        except ImportError:  # pragma: no cover
            self._vader = None
            self.method = "lexico_financiero"

    def score(self, text: str) -> float:
        """Sentimiento entre -1 (muy negativo) y +1 (muy positivo)."""
        if not text:
            return 0.0
        if self._vader is not None:
            return float(self._vader.polarity_scores(text)["compound"])
        palabras = re.findall(r"[a-z\-]+", text.lower())
        total = sum(FINANCIAL_LEXICON.get(p, 0.0) for p in palabras)
        return total / math.sqrt(total * total + 15) if total else 0.0


_ANALYZER: SentimentAnalyzer | None = None


def analyzer() -> SentimentAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentAnalyzer()
    return _ANALYZER
