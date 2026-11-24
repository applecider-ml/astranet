"""ASTRAnet domain plot style: fixed per-class colours for the 7 classes.

Validated for CVD separation and per-surface lightness/contrast on white and
#0d0d0d surfaces (light and dark are SEPARATE sets, not flips). Assign by class
name, never by position. Generic figure theming lives in `luna.style`.
"""
CLASS_ORDER = ["Ia", "Ibc", "SLSN", "II", "AGN", "CV", "TDE"]
CLASS_COLORS_LIGHT = {"Ia": "#e11d6a", "Ibc": "#f59e0b", "SLSN": "#a855f7",
                      "II": "#10b981", "AGN": "#0ea5e9", "CV": "#6366f1",
                      "TDE": "#9f1239"}
CLASS_COLORS_DARK = {"Ia": "#ec4899", "Ibc": "#d97706", "SLSN": "#a855f7",
                     "II": "#059669", "AGN": "#0284c7", "CV": "#6366f1",
                     "TDE": "#f43f5e"}


def class_colors(dark: bool = False) -> dict:
    return CLASS_COLORS_DARK if dark else CLASS_COLORS_LIGHT
