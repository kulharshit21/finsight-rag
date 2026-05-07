"""
XGBoost fraud pattern scorer.

Extracts financial signals from the LLM answer text and retrieved chunks,
then runs them through an XGBoost classifier trained on fraud patterns.

Signals extracted:
- revenue_growth_rate (YoY % change if mentioned)
- debt_to_equity_ratio
- cash_flow_mismatch (operating CF vs net income divergence)
- auditor_change_flag (True if different auditor mentioned)
- related_party_flag (True if related party transactions mentioned)
- goodwill_spike (True if goodwill increased >50%)

These are the exact same features used in ByteWatch — reuse that feature
engineering logic here.
"""

import re
import os
import numpy as np
from typing import List, Tuple, Optional
from langchain.schema import Document

from core.config import settings


FRAUD_FLAG_RULES = [
    # (regex pattern, flag description)
    (r"related.party", "Related party transactions detected"),
    (r"going.concern", "Going concern doubt mentioned"),
    (r"restatement|restated", "Financial restatement mentioned"),
    (r"qualified.opinion|adverse.opinion", "Auditor qualified/adverse opinion"),
    (r"material.weakness", "Material weakness in internal controls"),
    (r"goodwill.impairment", "Goodwill impairment mentioned"),
    (r"off.balance.sheet", "Off-balance sheet items detected"),
    (r"revenue.recognition", "Revenue recognition policy change"),
]


def extract_rule_based_flags(text: str) -> List[str]:
    """Fast rule-based red flag detection from document text."""
    text_lower = text.lower()
    flags = []
    for pattern, description in FRAUD_FLAG_RULES:
        if re.search(pattern, text_lower):
            flags.append(description)
    return flags


class FraudScorer:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load XGBoost model if available. Gracefully skip if not."""
        if not settings.fraud_model_path:
            return
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(settings.fraud_model_path)
        except Exception:
            self.model = None  # Model not available, use rule-based only

    def score(
        self,
        answer: str,
        docs_scores: List[Tuple[Document, float]],
    ) -> Tuple[Optional[float], List[str]]:
        """
        Returns (fraud_risk_score, flags).
        fraud_risk_score: 0–1 (None if XGBoost model not loaded)
        flags: list of human-readable red flags found
        """
        full_text = answer + " ".join(doc.page_content for doc, _ in docs_scores)

        # Rule-based flags (always run)
        flags = extract_rule_based_flags(full_text)

        # XGBoost score (only if model loaded)
        if self.model is None:
            # Heuristic score based on flag count
            score = min(len(flags) * 0.15, 1.0) if flags else 0.0
            return round(score, 3), flags

        # Feature extraction for XGBoost
        features = self._extract_features(full_text, flags)
        prob = self.model.predict_proba([features])[0][1]  # fraud class probability
        return round(float(prob), 3), flags

    def _extract_features(self, text: str, flags: List[str]) -> List[float]:
        """
        Extract numeric features for XGBoost.
        Reuse ByteWatch feature engineering here.
        """
        def find_ratio(pattern):
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    return 0.0
            return 0.0

        return [
            len(flags),                          # num_flags
            1.0 if "related party" in text.lower() else 0.0,
            1.0 if "going concern" in text.lower() else 0.0,
            1.0 if "restatement" in text.lower() else 0.0,
            1.0 if "qualified opinion" in text.lower() else 0.0,
            1.0 if "material weakness" in text.lower() else 0.0,
            find_ratio(r"debt.{1,30}(\d+\.?\d*)"),   # debt ratio if mentioned
            find_ratio(r"growth.{1,20}(\d+\.?\d*)"),  # growth rate if mentioned
            len(text) / 10000,                    # text length normalized
        ]
