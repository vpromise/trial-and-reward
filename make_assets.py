"""Writes assets/ for "Trial and Reward": the backdrop (deep plum, a soft vignette, a fine grid) and the base
icons — the "Agents, Animated" set recoloured into this palette, plus the grid world's charger, puddle and rock,
and a trophy, dice and a gamepad. Run once: python3 make_assets.py (needs numpy and Pillow)."""
import re
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
SOURCE = Path.home() / "videos" / "agent-series" / "assets"
ASSETS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------- backdrop
W, H = 1920, 1080
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
base = np.array([18, 14, 26], np.float32)            # #120E1A
lift = np.array([40, 30, 56], np.float32)            # a warm plum centre lift
r = np.sqrt(((x - W * 0.5) / (W * 0.62)) ** 2 + ((y - H * 0.45) / (H * 0.70)) ** 2)
t = np.clip(1 - r, 0, 1) ** 2
img = base + (lift - base) * t[..., None] * 0.55
img *= (1 - 0.35 * np.clip(r - 0.75, 0, 1))[..., None]
step = 1920 / 14.222 * 0.5                           # a line every half unit
minor = (np.abs((x + 0.5) % step) < 1.0) | (np.abs((y + 0.5) % step) < 1.0)
major = (np.abs((x + 0.5) % (step * 4)) < 1.2) | (np.abs((y + 0.5) % (step * 4)) < 1.2)
img += minor[..., None] * np.array([7, 5, 9], np.float32) * (0.6 + 0.4 * t[..., None])
img += major[..., None] * np.array([6, 5, 8], np.float32) * (0.6 + 0.4 * t[..., None])
Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(ASSETS / "backdrop.png")

# ---------------------------------------------------------------------------- recoloured base icons
GOLD, GOLD_FILL, BLUE, BLUE_FILL = "#FFC93C", "#3B2C0C", "#62A8FF", "#142849"
CORAL, CORAL_FILL, TEAL, TEAL_FILL = "#FF8A4C", "#3D1F10", "#34D6BD", "#0F322D"
VIOLET, VIOLET_FILL, PINK, PINK_FILL = "#B48CFF", "#2A1C47", "#F472D0", "#3D1535"
GREEN, GREEN_FILL, RED, RED_FILL = "#66E39A", "#133322", "#FF4D5E", "#3E141B"
REMAP = {
    "#FF5C7A": RED, "#3A1520": RED_FILL,                                          # rose → red
    "#3DDC97": GREEN, "#12301F": GREEN_FILL, "#1B6B47": "#1E6B44",                 # mint → green
    "#8EF0C4": "#A9F2C6", "#1B5A44": "#1B5538",
    "#FFC857": GOLD, "#3A2E14": GOLD_FILL, "#B7791F": "#B8861A", "#FFD98A": "#FFE08F",  # sun → gold
    "#5AB8FF": BLUE, "#132838": BLUE_FILL, "#D6ECFF": "#D6E7FF", "#4AA8F0": "#4C92F0",  # azure → blue
    "#C792EA": VIOLET, "#2C1F38": VIOLET_FILL, "#E7C4F5": "#DFD0FE", "#EEDDFB": "#EEE6FE",  # orchid → violet
    "#261B30": "#221933",
    "#FF7A59": CORAL,                                                             # coral → coral
    "#7A7166": "#7A6E8E", "#A9A094": "#A79DB8", "#DDD5C9": "#E2DCEC",            # warm greys → plum greys
    "#1E1B17": "#1C1726", "#14120F": "#120E1A",
}
for svg in sorted(SOURCE.glob("*.svg")):
    if re.match(r"ep\d", svg.name):
        continue
    out = re.sub(r"#[0-9A-Fa-f]{6}", lambda mt: REMAP.get(mt.group(0).upper(), mt.group(0)), svg.read_text())
    left = {c.upper() for c in re.findall(r"#[0-9A-Fa-f]{6}", out)} - {v.upper() for v in REMAP.values()}
    assert not left, (svg.name, left)
    (ASSETS / svg.name).write_text(out)

# ---------------------------------------------------------------------------- new icons (flat, 2-tone)
LT = "#E2DCEC"
NEW = {
    # the grid world: the charger (+1), the puddle (−1), the rock (a wall)
    "charger": f'''<rect x="18" y="14" width="64" height="72" rx="12" fill="{GOLD_FILL}" stroke="{GOLD}" stroke-width="5"/>
  <path d="M55 24 L36 54 H50 L44 76 L64 44 H50 Z" fill="{GOLD}"/>''',
    "puddle": f'''<path d="M50 14 C38 34 24 48 24 62 C24 77 36 88 50 88 C64 88 76 77 76 62 C76 48 62 34 50 14 Z"
  fill="{RED_FILL}" stroke="{RED}" stroke-width="5"/>
  <path d="M38 64 C38 72 43 77 50 78" fill="none" stroke="{RED}" stroke-width="5" stroke-linecap="round"/>''',
    "rock": f'''<path d="M14 74 L24 38 L44 22 L68 28 L86 52 L82 76 Z" fill="#2A2238" stroke="#A79DB8" stroke-width="5"
  stroke-linejoin="round"/><path d="M44 22 L50 48 L86 52 M50 48 L24 38 M50 48 L40 74" fill="none" stroke="#7A6E8E"
  stroke-width="4" stroke-linejoin="round"/>''',
    "trophy": f'''<path d="M30 14 H70 V40 C70 54 60 62 50 62 C40 62 30 54 30 40 Z" fill="{GOLD_FILL}" stroke="{GOLD}" stroke-width="5"/>
  <path d="M30 22 H16 C16 38 24 44 32 44 M70 22 H84 C84 38 76 44 68 44" fill="none" stroke="{GOLD}" stroke-width="5"/>
  <rect x="44" y="62" width="12" height="14" fill="{GOLD}"/><rect x="30" y="76" width="40" height="10" rx="3" fill="{GOLD}"/>''',
    "dice": f'''<rect x="16" y="16" width="68" height="68" rx="12" fill="{VIOLET_FILL}" stroke="{VIOLET}" stroke-width="5"/>
  {"".join(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{VIOLET}"/>' for cx, cy in ((34, 34), (66, 34), (50, 50), (34, 66), (66, 66)))}''',
    "gamepad": f'''<path d="M22 34 H78 C88 34 94 44 92 58 C90 72 84 80 76 78 C70 76 66 68 60 66 H40 C34 68 30 76 24 78
  C16 80 10 72 8 58 C6 44 12 34 22 34 Z" fill="{TEAL_FILL}" stroke="{TEAL}" stroke-width="5" stroke-linejoin="round"/>
  <path d="M26 46 V62 M18 54 H34" stroke="{TEAL}" stroke-width="5" stroke-linecap="round"/>
  <circle cx="70" cy="48" r="5" fill="{CORAL}"/><circle cx="78" cy="58" r="5" fill="{GOLD}"/>''',
    "robot-arm": f'''<rect x="12" y="78" width="40" height="10" rx="3" fill="#A79DB8"/>
  <path d="M32 78 L40 46 L70 30" fill="none" stroke="{CORAL}" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="40" cy="46" r="7" fill="{CORAL_FILL}" stroke="{CORAL}" stroke-width="4"/>
  <path d="M70 30 L84 22 M70 30 L82 40" stroke="{LT}" stroke-width="5" stroke-linecap="round"/>''',
}
for name, body in NEW.items():
    (ASSETS / f"{name}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n  {body}\n</svg>\n')
print(f"{len(list(ASSETS.glob('*.svg')))} icons + backdrop.png in {ASSETS}")
