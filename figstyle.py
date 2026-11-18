"""Dither / ASCII figure style shared across the enam4 repos.

How it works
------------
Charts are drawn with matplotlib with anti-aliasing switched off, so every pixel is an exact
palette colour. Fills use reserved "code colours" (see `D()`), which encode a ramp and a
density. `save()` swaps each code-coloured pixel for an ordered (Bayer 8x8) dither between the
background and that ramp's colour. Text and lines keep their solid colours, so they stay crisp.

Each repo picks its own palette with `P = theme("<name>")`. Categories are told apart by dither
density and direct labels rather than by hue, so the charts also work for colour-blind readers.
"""
import io
import os
import textwrap
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

# ------------------------------------------------------------------ palettes
# bg = background, fg = main text, dim = secondary text and grid, acc = the project's accent.
PALETTES = {
    "amber":   dict(bg="#120d06", fg="#ffe2b8", dim="#c09a66", acc="#ffa424"),   # qwen-vector-steering
    "cyan":    dict(bg="#04121a", fg="#d2f6ff", dim="#76aab6", acc="#2fd6f2"),   # sycophancy-probe
    "magenta": dict(bg="#14050f", fg="#ffd8f2", dim="#c084ad", acc="#ff4fc4"),   # qwen-glitch-tokens
    "violet":  dict(bg="#0d0818", fg="#e8dcff", dim="#9c8cc8", acc="#b48cff"),   # persona-mixing-steering
    "green":   dict(bg="#04110a", fg="#d0ffdb", dim="#7cb08a", acc="#48f57a"),   # emergent-misalignment
    "coral":   dict(bg="#140807", fg="#ffe0db", dim="#c48f86", acc="#ff6b57"),   # iweb
    "blue":    dict(bg="#050a17", fg="#dbe6ff", dim="#8a9fcc", acc="#5b8dff"),   # FileTree
    "yellow":  dict(bg="#121004", fg="#fff6c7", dim="#bfb27a", acc="#ffe04a"),   # Ideation-Skills
    "silver":  dict(bg="#0c0c0e", fg="#f0f0f2", dim="#a3a3ab", acc="#d9dce6"),   # KBsearch
    "lime":    dict(bg="#0b1004", fg="#eeffcc", dim="#a3b878", acc="#c4f23a"),   # ai-ready-knowledge-base
    "crimson": dict(bg="#14060a", fg="#ffdde3", dim="#c78a95", acc="#ff4766"),   # agent-injection-range
    "ice":     dict(bg="#060d14", fg="#e3f4ff", dim="#8fb0c4", acc="#a8e1ff"),   # jailbreak-decay-tracker
    "rose":    dict(bg="#140a10", fg="#ffe3ee", dim="#c4949f", acc="#ffa3c7"),   # refusal-direction
}
P = SimpleNamespace(**PALETTES["amber"])

WIDTH_IN, DPI = 10.0, 160      # 1600 px wide; GitHub shows it at roughly half size
CELL = 4                       # dither cell size in px, chunky enough to survive downscaling
MONO = ["DejaVu Sans Mono"]    # ships with matplotlib, so the look is the same everywhere
SCRIPT_FALLBACKS = ["Noto Sans Mono CJK JP", "Noto Sans CJK JP", "Microsoft YaHei", "Malgun Gothic",
                    "Yu Gothic", "Leelawadee UI", "Noto Sans Thai"]

_BAYER = np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
                   [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
                   [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                   [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]]) / 64 + 1 / 128
_RAMPS = ["acc", "fg", "dim"]
_CODE_R = 251  # red channel reserved for code colours; no palette colour uses it


def hexrgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def D(level, ramp="acc"):
    """A dithered fill: `level` 0..1 is the share of ramp-coloured pixels."""
    lv = int(round(max(0.0, min(1.0, level)) * 250)) + 2
    return "#%02x%02x%02x" % (_CODE_R, _RAMPS.index(ramp) + 2, lv)


def contrast(a, b):
    def lum(h):
        c = np.array(hexrgb(h)) / 255
        c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    la, lb = sorted([lum(a), lum(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def theme(name):
    """Select a palette and set up matplotlib. Returns the palette namespace."""
    P.__dict__.update(PALETTES[name])
    for k, minimum in (("fg", 7.0), ("dim", 4.5), ("acc", 4.5)):
        assert contrast(getattr(P, k), P.bg) >= minimum, (name, k, contrast(getattr(P, k), P.bg))
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": MONO + [n for n in SCRIPT_FALLBACKS if n in have],
        "font.size": 14,
        "text.antialiased": False, "lines.antialiased": False, "patch.antialiased": False,
        "axes.titlesize": 19, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 16, "axes.titlecolor": P.fg,
        "axes.labelsize": 14, "axes.labelcolor": P.dim,
        "axes.edgecolor": P.dim, "axes.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "grid.color": P.dim, "grid.linewidth": 1.0, "grid.linestyle": (0, (1, 5)),
        "axes.axisbelow": True,
        "xtick.color": P.dim, "ytick.color": P.dim, "xtick.labelcolor": P.dim, "ytick.labelcolor": P.dim,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "ytick.left": False,
        "xtick.major.size": 6, "xtick.major.width": 1.5,
        "text.color": P.fg,
        "figure.facecolor": P.bg, "axes.facecolor": P.bg, "savefig.facecolor": P.bg,
        "legend.frameon": False, "legend.fontsize": 13, "legend.labelcolor": P.fg,
        "legend.handlelength": 1.6, "legend.handleheight": 1.1,
        "lines.linewidth": 3, "lines.markersize": 9, "lines.solid_capstyle": "butt",
    })
    return P


def fig(h=5.6, left=0.09, right=0.97, top_in=1.25, bottom_in=0.85, **kw):
    """Fixed 1600 px wide canvas so every chart shows at the same scale on GitHub.
    Leave `top_in` inches for the title block and `bottom_in` for the x axis."""
    f, ax = plt.subplots(figsize=(WIDTH_IN, h), dpi=DPI, **kw)
    f.subplots_adjust(left=left, right=right, top=1 - top_in / h, bottom=bottom_in / h)
    return f, ax


def char_w(pt):
    """Width of one monospace character in inches."""
    return 0.602 * pt / 72


def title(ax, text, sub=None):
    """'> title' in the main colour, '# subtitle' in the dim colour underneath."""
    f = ax.figure; h = f.get_figheight()
    top = ax.get_position().y1
    head(f, text, sub, y_in=h - (top * h + 1.0))


def fit_pt(text, pt, width_in):
    """Largest font size <= pt at which `text` fits in `width_in` inches of monospace."""
    return min(pt, width_in / (len(text) * 0.602 / 72))


def head(f, text, sub=None, y_in=0.35):
    """Title block at the figure's left margin, `y_in` inches below the top edge."""
    h = f.get_figheight(); avail = WIDTH_IN * 0.91
    t = "> " + text
    f.text(0.045, 1 - y_in / h, t, color=P.fg, fontsize=fit_pt(t, 19, avail), fontweight="bold", va="top")
    if sub:
        s_ = "# " + sub
        f.text(0.045, 1 - (y_in + 0.48) / h, s_, color=P.dim, fontsize=fit_pt(s_, 12.5, avail), va="top")


def footer(f, text):
    f.text(0.03, 0.02, "-- " + text, color=P.dim, fontsize=11.5, ha="left", va="bottom")


def label(ax, x, y, text, color=None, boxed=False, **kw):
    """Direct label. `boxed=True` puts it on a solid background so it reads over a dither."""
    kw.setdefault("fontsize", 13); kw.setdefault("fontweight", "bold")
    if boxed:
        kw.setdefault("bbox", dict(facecolor=P.bg, edgecolor="none", boxstyle="square,pad=0.15"))
    return ax.text(x, y, text, color=color or P.fg, **kw)


def _snap(a):
    """Map every pixel to the nearest palette colour (removes stray blended pixels)."""
    pal = np.array([hexrgb(c) for c in (P.bg, P.fg, P.dim, P.acc)], dtype=np.int32)
    flat = a.reshape(-1, 3).astype(np.int32)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    near = pal[((uniq[:, None, :] - pal[None]) ** 2).sum(-1).argmin(1)]
    return near[inv.ravel()].reshape(a.shape).astype(np.uint8)


def _dither(img):
    a = np.asarray(img.convert("RGB")).copy()
    mask = a[..., 0] == _CODE_R
    if mask.any():
        ramp = np.clip(a[..., 1].astype(int) - 2, 0, len(_RAMPS) - 1)
        level = (a[..., 2].astype(float) - 2) / 250
        yy, xx = np.indices(mask.shape)
        thr = _BAYER[(yy // CELL) % 8, (xx // CELL) % 8]
        on = level > thr
        cols = np.array([hexrgb(getattr(P, r)) for r in _RAMPS], dtype=np.uint8)
        bg = np.array(hexrgb(P.bg), dtype=np.uint8)
        a[mask & on] = cols[ramp[mask & on]]
        a[mask & ~on] = bg
    return Image.fromarray(_snap(a))


def save(f, name, out="assets", tight=False):
    os.makedirs(out, exist_ok=True)
    buf = io.BytesIO()
    f.savefig(buf, format="png", dpi=DPI, bbox_inches="tight" if tight else None, pad_inches=0.35)
    plt.close(f)
    img = _dither(Image.open(buf))
    p = os.path.join(out, name)
    img.convert("P", palette=Image.ADAPTIVE, colors=16).save(p, optimize=True)
    print("wrote", p, img.size)


def dbar(ax, x, h, level, ramp="acc", horizontal=False, edge=True, **kw):
    """Bar with a dithered fill and a solid outline in the ramp colour."""
    ec = getattr(P, ramp) if edge else "none"
    fn = ax.barh if horizontal else ax.bar
    return fn(x, h, color=D(level, ramp), edgecolor=ec, linewidth=2 if edge else 0, **kw)


def swatch(level, ramp="acc", text=""):
    from matplotlib.patches import Patch
    return Patch(facecolor=D(level, ramp), edgecolor=getattr(P, ramp), linewidth=2, label=text)


def terminal(name, title_text, sub, rows, label_w=10, foot=None, levels=None):
    """Example outputs as a terminal transcript.

    rows = [(label, text)]; levels = optional 0..1 per row, drawn as a dithered gutter bar
    (for example, push strength)."""
    x0 = 0.62 + label_w * char_w(13) + 0.25
    wrap = int((WIDTH_IN - x0 - 0.3) / char_w(12.5))
    wrapped = [textwrap.wrap(t, wrap) or [""] for _, t in rows]
    lh = 0.27
    heights = [0.16 + lh * len(w) for w in wrapped]
    H = 1.35 + sum(heights) + 0.14 * len(rows) + (0.45 if foot else 0.1)
    f = plt.figure(figsize=(WIDTH_IN, H), dpi=DPI)
    ax = f.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, WIDTH_IN); ax.set_ylim(0, H)
    y = H - 0.32
    ax.text(0.3, y, "> " + title_text, fontsize=fit_pt("> " + title_text, 18, WIDTH_IN - 0.6),
            fontweight="bold", va="top", color=P.fg)
    y -= 0.42
    ax.text(0.3, y, "# " + sub, fontsize=fit_pt("# " + sub, 12.5, WIDTH_IN - 0.6), color=P.dim, va="top")
    y -= 0.38
    ax.plot([0.3, WIDTH_IN - 0.3], [y - 0.08, y - 0.08], color=P.dim, lw=1.5)
    y -= 0.26
    from matplotlib.patches import Rectangle
    for i, ((lab, _), w, h) in enumerate(zip(rows, wrapped, heights)):
        lv = levels[i] if levels is not None else 1.0
        ax.add_patch(Rectangle((0.3, y - h + 0.06), 0.16, h - 0.04, facecolor=D(max(lv, 0.12)), edgecolor="none"))
        ax.text(0.62, y - 0.04, lab, fontsize=13, fontweight="bold", color=P.acc, va="top")
        for j, line in enumerate(w):
            ax.text(x0, y - 0.04 - j * lh, line, fontsize=12.5, va="top", color=P.fg, parse_math=False)
        y -= h + 0.14
    if foot:
        ax.text(0.3, 0.2, "-- " + textwrap.shorten(foot, int((WIDTH_IN - 0.6) / char_w(11.5)) - 3, placeholder=" …"),
                fontsize=11.5, color=P.dim, va="bottom")
    save(f, name, tight=False)


def banner(name, repo, tagline, field, out="assets", size=(1600, 400),
           meta="> qwen2.5 · cpu only · github.com/enam4"):
    """Wide dithered header: a 0..1 `field` (rows x cols array) dithered in the accent colour, with
    the repo name and tagline set in a dark box. size=(1280, 640) gives a GitHub social preview."""
    W, H = size
    f = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = f.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, W); ax.set_ylim(0, H)
    cells = np.clip(np.asarray(field, dtype=float), 0, 1)
    ry, rx = -(-H // cells.shape[0]), -(-W // cells.shape[1])
    lv = np.repeat(np.repeat(cells, ry, axis=0), rx, axis=1)[:H, :W]
    code = np.zeros((H, W, 3), dtype=np.uint8)
    code[..., 0] = _CODE_R; code[..., 1] = 2; code[..., 2] = (np.round(lv * 250) + 2).astype(np.uint8)
    ax.imshow(code, extent=(0, W, 0, H), interpolation="nearest", aspect="auto")
    from matplotlib.patches import Rectangle
    bw, cy = min(1060, W - 96), H / 2
    ax.add_patch(Rectangle((48, cy - 148), bw, 296, facecolor=P.bg, edgecolor=P.dim, linewidth=1.5))
    ax.text(84, cy + 112, repo, fontsize=fit_pt(repo, 30, (bw - 80) / DPI), fontweight="bold", color=P.fg, va="top")
    assert len(textwrap.wrap(tagline, 52)) <= 2, "keep the banner tagline to two lines"
    ax.text(84, cy + 22, textwrap.fill(tagline, 52), fontsize=13.5, color=P.dim, va="top", linespacing=1.55)
    ax.text(84, cy - 100, meta, fontsize=12, color=P.acc, va="top")
    os.makedirs(out, exist_ok=True)
    buf = io.BytesIO(); f.savefig(buf, format="png", dpi=DPI); plt.close(f)
    img = _dither(Image.open(buf)).convert("P", palette=Image.ADAPTIVE, colors=16)
    p = os.path.join(out, name); img.save(p, optimize=True); print("wrote", p, img.size)


def term_capture(name, cmd, text, out="assets"):
    """Render real terminal output (with ANSI colours) as a panel in the current palette.
    ANSI red and cyan map to the accent, yellow and bold to the main colour, green and grey to dim."""
    import re as _re
    from PIL import ImageDraw, ImageFont
    fdir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    role = {31: "acc", 91: "acc", 36: "acc", 96: "acc", 33: "fg", 93: "fg", 32: "dim", 92: "dim", 90: "dim", 37: "fg", 97: "fg"}
    lines = []
    for raw in text.replace("\r", "").rstrip("\n").split("\n"):
        segs, col, bold = [], "fg", False
        for part in _re.split(r"(\x1b\[[0-9;]*m)", raw):
            m = _re.match(r"\x1b\[([0-9;]*)m", part)
            if m:
                for c in (m.group(1) or "0").split(";"):
                    c = int(c or 0)
                    if c == 0:
                        col, bold = "fg", False
                    elif c == 1:
                        bold = True
                    elif c in role:
                        col = role[c]
            elif part:
                segs.append((part, col, bold))
        lines.append(segs)
    longest = max([len("".join(t for t, _, _ in l)) for l in lines] + [len(cmd) + 2])
    Wpx, pad = 1600, 48
    size = int(min(26, (Wpx - 2 * pad) / (longest * 0.602)))
    f = ImageFont.truetype(os.path.join(fdir, "DejaVuSansMono.ttf"), size)
    fb = ImageFont.truetype(os.path.join(fdir, "DejaVuSansMono-Bold.ttf"), size)
    cw, lh, top = size * 0.602, int(size * 1.45), 92
    Hpx = top + lh * (len(lines) + 1) + pad
    # a dithered strip along the top, like a window bar, then the text on the plain background
    code = np.zeros((Hpx, Wpx, 3), dtype=np.uint8); code[:] = hexrgb(P.bg)
    lv = np.clip(0.55 - 0.45 * np.arange(Wpx)[None, :] / Wpx, 0, 1) * np.ones((44, 1))
    code[:44, :, 0] = _CODE_R; code[:44, :, 1] = 2; code[:44, :, 2] = (np.round(lv * 250) + 2).astype(np.uint8)
    img = _dither(Image.fromarray(code))
    d = ImageDraw.Draw(img); d.fontmode = "1"
    d.rectangle((pad - 12, 6, pad + cw * (len(cmd) + 3) + 12, 38), fill=P.bg)
    d.text((pad, 10), "> " + cmd, font=fb, fill=P.acc)
    y = top
    for segs in lines:
        x = pad
        for t, c, b in segs:
            d.text((x, y), t, font=fb if b else f, fill=getattr(P, c)); x += cw * len(t)
        y += lh
    d.text((pad, y), "> █", font=f, fill=P.dim)
    os.makedirs(out, exist_ok=True)
    img = Image.fromarray(_snap(np.asarray(img.convert("RGB")).copy()))
    p = os.path.join(out, name); img.convert("P", palette=Image.ADAPTIVE, colors=8).save(p, optimize=True)
    print("wrote", p, img.size)
