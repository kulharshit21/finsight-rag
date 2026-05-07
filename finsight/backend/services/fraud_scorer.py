"""
Financial red-flag detector and fraud risk scorer.

Two layers run on every query response:

Layer 1 — Rule-based (always active)
  Regex patterns scan both the LLM answer and retrieved chunks for language
  that frequently signals financial distress or manipulation:
  going concern, restatements, related-party transactions, qualified opinions,
  material weaknesses, off-balance-sheet items, revenue recognition changes.

Layer 2 — XGBoost classifier (optional, loads from fraud_model_path)
  If a trained model file is present it takes numeric features extracted from
  the same text and outputs a calibrated fraud probability 0–1.
  If no model file is present, a simple heuristic (0.15 per flag, capped at 1)
  is used so the application still runs meaningfully without a trained model.
"""

import re
from typing import List, Optional, Tuple

from langchain.schema import Document

from core.config import settings


# Each tuple: (regex pattern, human-readable flag description)
FRAUD_FLAG_RULES: List[Tuple[str, str]] = [
    (r"related.party",              "Related party transactions detected"),
    (r"going.concern",              "Going concern doubt mentioned"),
    (r"restatement|restated",       "Financial restatement mentioned"),
    (r"qualified.opinion|adverse.opinion", "Auditor qualified/adverse opinion"),
    (r"material.weakness",          "Material weakness in internal controls"),
    (r"goodwill.impairment",        "Goodwill impairment mentioned"),
    (r"off.balance.sheet",          "Off-balance-sheet items detected"),
    (r"revenue.recognition",        "Revenue recognition policy change"),
]


def _extract_flags(text: str) -> List[str]:
    t = text.lower()
    return [desc for pattern, desc in FRAUD_FLAG_RULES if re.search(pattern, t)]


class FraudScorer:
    def __init__(self):
        self.model = None
        self._try_load_model()

    def _try_load_model(self):
        if not settings.fraud_model_path:
            return
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(settings.fraud_model_path)
        except Exception:
            self.model = None   # Gracefully degrade to heuristic scoring

    def score(
        self,
        answer: str,
        docs_scores: List[Tuple[Document, float]],
    ) -> Tuple[Optional[float], List[str]]:
        """
        Returns (fraud_risk_score 0–1, triggered_flags).
        Score is None only when model loading itself failed mid-run.
        """
        full_text = answer + " " + " ".join(d.page_content for d, _ in docs_scores)
        flags = _extract_flags(full_text)

        if self.model is None:
            # Heuristic: each flag adds 0.15 risk, capped at 1.0
            score = round(min(len(flags) * 0.15, 1.0), 3)
            return score, flags

        features = self._features(full_text, flags)
        prob = self.model.predict_proba([features])[0][1]
        return round(float(prob), 3), flags

    def _features(self, text: str, flags: List[str]) -> List[float]:
        """Numeric feature vector for the XGBoost model."""
        t = text.lower()

        def find_num(pattern: str) -> float:
            m = re.search(pattern, t)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return 0.0
            return 0.0

        return [
            len(flags),
            1.0 if "related party" in t else 0.0,
            1.0 if "going concern" in t else 0.0,
            1.0 if "restatement" in t else 0.0,
            1.0 if "qualified opinion" in t else 0.0,
            1.0 if "material weakness" in t else 0.0,
            find_num(r"debt.{1,30}(\d+\.?\d*)"),
            find_num(r"growth.{1,20}(\d+\.?\d*)"),
            len(text) / 10_000,
        ]
