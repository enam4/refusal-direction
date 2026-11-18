# render real command output (with ANSI colours) as a terminal-window PNG.
# window title "llmvuln", prompt "llmvuln@lab:~/experiments$". no host, user or path from the machine appears.
import re, subprocess, sys, os
from PIL import Image, ImageDraw, ImageFont
F = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"; FB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
PAL = {30: (46, 52, 54), 31: (204, 0, 0), 32: (78, 154, 6), 33: (196, 160, 0), 34: (52, 101, 164), 35: (117, 80, 123), 36: (6, 152, 154), 37: (211, 215, 207),
       90: (85, 87, 83), 91: (239, 41, 41), 92: (138, 226, 52), 93: (252, 233, 79), 94: (114, 159, 207), 95: (173, 127, 168), 96: (52, 226, 226), 97: (238, 238, 236)}
FG = (230, 230, 230); BG = (0, 0, 0)
PROMPT = [("llmvuln@lab", (138, 226, 52), True), (":", FG, False), ("~/experiments", (114, 159, 207), True), ("$ ", FG, False)]

def parse(line):
    segs = []; col = FG; bold = False
    for part in re.split(r"(\x1b\[[0-9;]*m)", line):
        m = re.match(r"\x1b\[([0-9;]*)m", part)
        if m:
            for c in (m.group(1) or "0").split(";"):
                c = int(c or 0)
                if c == 0: col, bold = FG, False
                elif c == 1: bold = True
                elif c in PAL: col = PAL[c]
        elif part:
            segs.append((part, col, bold))
    return segs

def render(cmd, text, out, width=1240, size=18):
    f = ImageFont.truetype(F, size); fb = ImageFont.truetype(FB, size)
    lines = [PROMPT + [(cmd, FG, False)]] + [parse(l) for l in text.rstrip("\n").split("\n")] + [PROMPT + [("\u2588", FG, False)]]
    lh = int(size * 1.35); top = 40
    H = top + 14 + lh * len(lines) + 14
    im = Image.new("RGB", (width, H), BG); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, width, top - 6], fill=(238, 238, 236)); d.line([0, top - 6, width, top - 6], fill=(190, 190, 190))
    tf = ImageFont.truetype(F, 15); tw = d.textlength("llmvuln", font=tf)
    d.text(((width - tw) / 2, 8), "llmvuln", font=tf, fill=(60, 60, 60))
    for i, s in enumerate(["\u2013", "\u25a1", "\u00d7"]): d.text((width - 80 + i * 24, 6), s, font=tf, fill=(60, 60, 60))
    y = top + 8
    for segs in lines:
        x = 10
        for t, c, b in segs:
            ff = fb if b else f
            d.text((x, y), t, font=ff, fill=c); x += d.textlength(t, font=ff)
        y += lh
    im.save(out)  # PIL writes no text chunks unless asked; checked again by the clean-up audit

if __name__ == "__main__":
    # usage: python term.py out.png "shown command" real_command...
    out, shown = sys.argv[1], sys.argv[2]
    env = dict(os.environ, FORCE_COLOR="1")
    txt = subprocess.run(sys.argv[3:], capture_output=True, text=True, env=env).stdout
    render(shown, txt, out)
    print("wrote", out)
