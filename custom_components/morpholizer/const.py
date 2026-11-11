"""Constants for Morpholizer."""

from typing import Final

DOMAIN: Final = "morpholizer"
CONF_FALLBACK_AGENT: Final = "fallback_agent"
ENTITY_NAME: Final = "Morpholizer"
CONF_FUZZY_ENABLED: Final = "fuzzy_enabled"
CONF_FUZZY_SCORE_CUTOFF: Final = "fuzzy_score_cutoff"
CONF_FUZZY_MARGIN: Final = "fuzzy_margin"
FUZZY_DEFAULT_ENABLED: Final = False
FUZZY_DEFAULT_SCORE_CUTOFF: Final = 90.0
FUZZY_DEFAULT_MARGIN: Final = 8.0
FUZZY_MIN_SCORE_CUTOFF: Final = 85.0
FUZZY_MAX_SCORE_CUTOFF: Final = 100.0
FUZZY_MIN_MARGIN: Final = 5.0
FUZZY_MAX_MARGIN: Final = 30.0
