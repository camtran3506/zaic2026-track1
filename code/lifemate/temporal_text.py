"""Shared finite Vietnamese number/clock grammar for runtime and SFT selectors."""
import re


UNITS = {"không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
         "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9}
_UNIT_PATTERN = "(?:" + "|".join(UNITS) + ")"
NUMBER = rf"(?:\d{{1,2}}|mười(?:\s+{_UNIT_PATTERN})?|{_UNIT_PATTERN}\s+mươi(?:\s+{_UNIT_PATTERN})?|{_UNIT_PATTERN})"
CLOCK = re.compile(rf"(?<!\w)(?P<hour>{NUMBER})\s*(?:(?P<half>rưỡi)|(?:giờ|:|h)\s*(?P<minute>{NUMBER})?)(?!\w)")


def number(value):
    if value.isdigit():
        return int(value)
    words = value.split()
    if words[0] == "mười":
        return 10 + (UNITS[words[1]] if len(words) == 2 else 0)
    if len(words) > 1 and words[1] == "mươi":
        return UNITS[words[0]] * 10 + (UNITS[words[2]] if len(words) == 3 else 0)
    return UNITS[words[0]]
