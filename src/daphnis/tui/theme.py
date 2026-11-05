"""Colors shared by terminal widgets and Rich renderables."""

from textual.theme import Theme

BACKGROUND = "#0d121b"
SURFACE = "#171e2a"
BORDER = "#48566b"
TEXT = "#e7edf7"
MUTED = "#9aacc6"
ACCENT = "#a89bff"
BLUE = "#7aa2ff"
SUCCESS = "#7ee6a4"
WARNING = "#e3c77b"
ERROR = "#ed8298"

THEME = Theme(
    name="daphnis",
    primary=ACCENT,
    secondary=BLUE,
    accent=ACCENT,
    foreground=TEXT,
    background=BACKGROUND,
    surface=SURFACE,
    panel="#242e41",
    success=SUCCESS,
    warning=WARNING,
    error=ERROR,
    variables={
        "block-cursor-foreground": BACKGROUND,
        "block-cursor-background": ACCENT,
        "block-cursor-text-style": "bold",
        "input-selection-background": "#a89bff 25%",
        "button-color-foreground": BACKGROUND,
    },
)
