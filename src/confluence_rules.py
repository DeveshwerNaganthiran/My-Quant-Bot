from typing import Tuple


def evaluate_confluence_rule(
    smc_confidence: float,
    ai_confidence: float,
    ai_signal: str,
    smc_signal_type: str,
) -> Tuple[bool, str]:
    """Return whether a trade should be allowed based on SMC/AI confluence.

    Rules:
    - SMC must be more than 75%
    - If AI confidence is above 80% OR total confidence is above 80%, allow direct execution
    - If AI confidence is between 60% and 80% or total confidence is between 60% and 80%,
      allow the setup through the normal filter path but do not force direct execution
    - If either side is below 60%, reject the setup
    """
    if ai_signal not in {"BUY", "SELL"} or smc_signal_type not in {"BUY", "SELL"}:
        return False, "signal is not a tradable direction"

    if ai_signal != smc_signal_type:
        return False, "AI and SMC disagree"

    if smc_confidence <= 0.75:
        return False, f"SMC confidence too low ({smc_confidence:.0%}, need >75%)"

    if ai_confidence < 0.60:
        return False, f"AI confidence too low ({ai_confidence:.0%}, need ≥60%)"

    total_confidence = (smc_confidence + ai_confidence) / 2.0

    if ai_confidence > 0.80 or total_confidence > 0.80:
        return True, f"direct execution allowed (AI {ai_confidence:.0%}, total {total_confidence:.0%})"

    if ai_confidence >= 0.60 or total_confidence >= 0.60:
        return True, f"normal filter path (AI {ai_confidence:.0%}, total {total_confidence:.0%})"

    return False, f"confidence below required band (AI {ai_confidence:.0%}, total {total_confidence:.0%})"
