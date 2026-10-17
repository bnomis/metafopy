"""Build the metafopy logo files in logo/: SVGs (wordmark outlined), PNGs, favicons and the animated mark.

The mark is a Baujat plot: six studies in a cluster and one influential study (the large ball), in a white rounded box with a
hairline graphite border. Dots use the Solarized accents (https://ethanschoonover.com/solarized/); the wordmark is Source Serif 4
Display Semibold (https://github.com/adobe-fonts/source-serif, SIL Open Font License), converted to outlines.

The animated mark bounces the dots around the box, slows them down and settles them in their final positions. Its random
generator is mulberry32, the same one the canvas preview uses in JavaScript, so both show the same bounce.

PNGs and GIF frames are rendered with headless Chromium.

Usage (dependencies are declared inline, PEP 723):
    uv run --script bin/logo.py --font SourceSerif4Display-Semibold.otf --chrome /path/to/chromium
"""

# /// script
# requires-python = ">=3.14"
# dependencies = ["fonttools", "pillow", "uharfbuzz"]
# ///

import argparse
import collections.abc
import math
import pathlib
import subprocess
import tempfile

import fontTools.pens.svgPathPen
import fontTools.pens.transformPen
import fontTools.ttLib
import PIL.Image
import uharfbuzz

ROOT = pathlib.Path(__file__).resolve().parents[1]

GRAPHITE = '#3D3F44'
# Border width on the 180-unit box: 0.5 px when the mark is drawn 180 px wide.
STROKE = 0.5
# Mark coordinates are on a 100-unit grid, placed in the 180-unit box by translate(24) scale(1.32).
MARK = [
    (20, 80, 5.5, '#CB4B16'),
    (32, 74, 5.5, '#859900'),
    (22, 64, 5.5, '#268BD2'),
    (40, 84, 5.5, '#B58900'),
    (38, 62, 5.5, '#6C71C4'),
    (48, 74, 5.5, '#2AA198'),
    (76, 26, 12, '#DC322F'),
]
BALLS = [(24 + 1.32 * x, 24 + 1.32 * y, 1.32 * r, colour) for x, y, r, colour in MARK]

FPS = 60
SETTLE_START = 1.6
DURATION = 3.2
HOLD = 4.0
INSET = 9.0
SEED = 7

type Frame = list[tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='logo.py',
        description='Build the metafopy logo files.',
    )
    parser.add_argument(
        '--chrome',
        default='chromium',
        help='Headless Chromium (or chrome-headless-shell) used to render PNGs',
    )
    parser.add_argument(
        '--font',
        type=pathlib.Path,
        required=True,
        help='SourceSerif4Display-Semibold.otf from https://github.com/adobe-fonts/source-serif/releases',
    )
    parser.add_argument(
        '--out',
        type=pathlib.Path,
        default=ROOT / 'logo',
        help='Output directory',
    )
    return parser.parse_args()


def box(stroke: float) -> str:
    half = stroke / 2
    side = 180 - stroke
    return f'<rect x="{half:g}" y="{half:g}" width="{side:g}" height="{side:g}" rx="40" fill="#FFFFFF" stroke="{GRAPHITE}" stroke-width="{stroke:g}"/>'


def mark_body(stroke: float) -> str:
    circles = ''.join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{colour}"/>' for x, y, r, colour in BALLS)
    return box(stroke) + circles


def mark_svg(stroke: float) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 180" width="180" height="180" role="img" aria-label="metafopy">{mark_body(stroke)}</svg>\n'
    )


def wordmark_path(font_path: pathlib.Path, text: str, size: float, x0: float, baseline: float) -> tuple[str, float]:
    face = uharfbuzz.Face(uharfbuzz.Blob.from_file_path(str(font_path)))
    font = uharfbuzz.Font(face)
    buf = uharfbuzz.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    uharfbuzz.shape(font, buf, {'kern': True, 'liga': True})
    tt = fontTools.ttLib.TTFont(str(font_path))
    glyph_set = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    scale = size / tt['head'].unitsPerEm
    pen = fontTools.pens.svgPathPen.SVGPathPen(glyph_set)
    cursor = 0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
        offset = (x0 + (cursor + pos.x_offset) * scale, baseline - pos.y_offset * scale)
        glyph_set[order[info.codepoint]].draw(fontTools.pens.transformPen.TransformPen(pen, (scale, 0, 0, -scale, *offset)))
        cursor += pos.x_advance
    return pen.getCommands(), cursor * scale


def logo_svg(font_path: pathlib.Path) -> str:
    mark = 96
    gap = 18
    size = 56
    line = 64
    tt = fontTools.ttLib.TTFont(str(font_path))
    upm = tt['head'].unitsPerEm
    ascent = tt['hhea'].ascent * size / upm
    descent = -tt['hhea'].descent * size / upm
    baseline = (mark - line) / 2 + (line - (ascent + descent)) / 2 + ascent
    path, width = wordmark_path(font_path, 'metafopy', size, mark + gap, baseline)
    total = mark + gap + width
    scale = mark / 180
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total:.2f} {mark}" width="{total:.2f}" height="{mark}" role="img" aria-label="metafopy">'
        f'<g transform="scale({scale:g})">{mark_body(STROKE / scale)}</g>'
        f'<path d="{path}" fill="{GRAPHITE}"/></svg>\n'
    )


def render(chrome: str, svg: str, png: pathlib.Path, width: int, height: int, background: str = 'transparent') -> None:
    inline = svg.replace('<svg ', '<svg style="display:block;width:100%;height:100%" ', 1)
    with tempfile.TemporaryDirectory() as tmp:
        html = pathlib.Path(tmp) / 'page.html'
        html.write_text(
            f'<!doctype html><html><body style="margin:0;background:{background}"><div style="width:{width}px;height:{height}px">{inline}</div></body></html>'
        )
        subprocess.run(
            [
                chrome,
                '--headless',
                '--no-sandbox',
                '--disable-gpu',
                '--hide-scrollbars',
                '--force-device-scale-factor=1',
                '--default-background-color=00000000',
                '--virtual-time-budget=1000',
                f'--window-size={width},{height}',
                f'--screenshot={png}',
                html.as_uri(),
            ],
            check=True,
            capture_output=True,
        )


def mulberry32(seed: int) -> collections.abc.Callable[[], float]:
    state = seed & 0xFFFFFFFF

    def imul(x: int, y: int) -> int:
        return (x * y) & 0xFFFFFFFF

    def next_float() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(state ^ (state >> 15), 1 | state)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return next_float


def smoothstep(a: float, b: float, t: float) -> float:
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    u = (t - a) / (b - a)
    return u * u * u * (u * (6 * u - 15) + 10)


def simulate() -> list[Frame]:
    rand = mulberry32(SEED)

    def uniform(lo: float, hi: float) -> float:
        return lo + (hi - lo) * rand()

    state = []
    for _, _, r, _ in BALLS:
        angle = uniform(0, 2 * math.pi)
        speed = uniform(260, 380)
        state.append([uniform(INSET + r, 180 - INSET - r), uniform(INSET + r, 180 - INSET - r), speed * math.cos(angle), speed * math.sin(angle)])
    frames = []
    dt = 1 / FPS
    damping = math.exp(-1.1 * dt)
    for i in range(round(DURATION * FPS) + 1):
        blend = smoothstep(SETTLE_START, DURATION, i * dt)
        frames.append([(s[0] + (fx - s[0]) * blend, s[1] + (fy - s[1]) * blend) for s, (fx, fy, _, _) in zip(state, BALLS, strict=True)])
        for s, (_, _, r, _) in zip(state, BALLS, strict=True):
            s[0] += s[2] * dt
            s[1] += s[3] * dt
            lo, hi = INSET + r, 180 - INSET - r
            if not lo <= s[0] <= hi:
                s[0] = min(max(s[0], lo), hi)
                s[2] = -s[2] * 0.9
            if not lo <= s[1] <= hi:
                s[1] = min(max(s[1], lo), hi)
                s[3] = -s[3] * 0.9
            s[2] *= damping
            s[3] *= damping
    return frames


def animated_svg(frames: list[Frame]) -> str:
    total = DURATION + HOLD
    key_times = ';'.join([f'{i / (len(frames) - 1) * DURATION / total:.4f}' for i in range(len(frames))] + ['1'])
    circles = []
    for k, (fx, fy, r, colour) in enumerate(BALLS):
        xs = ';'.join(f'{f[k][0]:.2f}' for f in frames) + f';{fx:.2f}'
        ys = ';'.join(f'{f[k][1]:.2f}' for f in frames) + f';{fy:.2f}'
        circles.append(
            f'<circle cx="{fx:.2f}" cy="{fy:.2f}" r="{r:.2f}" fill="{colour}">'
            f'<animate attributeName="cx" dur="{total}s" repeatCount="indefinite" values="{xs}" keyTimes="{key_times}"/>'
            f'<animate attributeName="cy" dur="{total}s" repeatCount="indefinite" values="{ys}" keyTimes="{key_times}"/>'
            '</circle>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 180" width="180" height="180" role="img" aria-label="metafopy">'
        f'{box(STROKE)}{"".join(circles)}</svg>\n'
    )


def frame_svg(positions: Frame) -> str:
    circles = ''.join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{c}"/>' for (x, y), (_, _, r, c) in zip(positions, BALLS, strict=True))
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 180">{box(STROKE)}{circles}</svg>'


def animated_gif(chrome: str, frames: list[Frame], gif: pathlib.Path, px: int, fps: int) -> None:
    picked = frames[:: FPS // fps]
    if picked[-1] is not frames[-1]:
        picked.append(frames[-1])
    images = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, positions in enumerate(picked):
            png = pathlib.Path(tmp) / f'frame-{i:03d}.png'
            render(chrome, frame_svg(positions), png, px, px, background='#FFFFFF')
            images.append(PIL.Image.open(png).convert('RGB'))
    durations = [round(1000 / fps)] * len(images)
    durations[-1] = round(HOLD * 1000)
    images[0].save(gif, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)


def main() -> None:
    args = parse_args()
    out = args.out
    for sub in ('favicon', 'png', 'svg'):
        (out / sub).mkdir(parents=True, exist_ok=True)

    mark = mark_svg(STROKE)
    logo = logo_svg(args.font)
    (out / 'svg' / 'metafopy-mark.svg').write_text(mark)
    (out / 'svg' / 'metafopy-logo.svg').write_text(logo)
    for px in (64, 128, 256, 512, 1024):
        render(args.chrome, mark, out / 'png' / f'metafopy-mark-{px}.png', px, px)
    logo_width = float(logo.split('viewBox="0 0 ')[1].split(' ')[0])
    for scale in (1, 2, 4):
        render(args.chrome, logo, out / 'png' / f'metafopy-logo@{scale}x.png', round(logo_width * scale), 96 * scale)

    # Favicons keep the border at 0.5 px whatever their size.
    (out / 'favicon' / 'favicon.svg').write_text(mark_svg(180 * 0.5 / 32))
    for px in (16, 32, 48):
        render(args.chrome, mark_svg(180 * 0.5 / px), out / 'favicon' / f'favicon-{px}.png', px, px)
    render(args.chrome, mark, out / 'favicon' / 'apple-touch-icon.png', 180, 180)
    icons = [PIL.Image.open(out / 'favicon' / f'favicon-{px}.png') for px in (16, 32, 48)]
    icons[2].save(out / 'favicon' / 'favicon.ico', sizes=[(16, 16), (32, 32), (48, 48)], append_images=icons[:2])

    frames = simulate()
    (out / 'svg' / 'metafopy-mark-animated.svg').write_text(animated_svg(frames))
    animated_gif(args.chrome, frames, out / 'png' / 'metafopy-mark-animated-256.gif', 256, 30)
    print(f'wrote {len([p for p in out.rglob("*") if p.is_file()])} files to {out}')


if __name__ == '__main__':
    main()
