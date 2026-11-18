# shared look for llmvuln cards (dark, github-like palette, DejaVu Sans Mono)
from PIL import Image, ImageDraw, ImageFont
F = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"; FB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
def font(sz, bold=False): return ImageFont.truetype(FB if bold else F, sz)
BG = (13, 17, 23); FG = (201, 209, 217); DIM = (125, 133, 144); GRN = (63, 185, 80); RED = (248, 81, 73); CY = (88, 166, 255)
PAN = (22, 27, 34); BOR = (48, 54, 61); YEL = (210, 153, 34); MAG = (188, 140, 255); GRY = (72, 79, 88); ORG = (230, 120, 60)
def canvas(W, H, title):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 46], fill=PAN)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]): d.ellipse([20 + i * 26, 15, 36 + i * 26, 31], fill=c)
    d.text((110, 13), title, font=font(17), fill=DIM)
    return im, d
def bar(d, x, y, w, h, frac, col):
    d.rounded_rectangle([x, y, x + w, y + h], 6, fill=(33, 38, 45))
    if frac > 0: d.rounded_rectangle([x, y, x + max(8, int(w * frac)), y + h], 6, fill=col)
def save(im, path):
    im.save(path, optimize=True)  # no text chunks are written
