/*
 * metafopy favicon: the large dot (the influential study) fades slowly between colours.
 *
 * Include once per page, after the static favicon links:
 *   <link rel="icon" href="favicon.svg" type="image/svg+xml">
 *   <script src="favicon-animate.js" defer></script>
 *
 * Draws the mark on a canvas and swaps the icon's href for a PNG data URL a few times a second.
 * Works in Chrome, Edge and Firefox; Safari keeps the static icon. Respects prefers-reduced-motion
 * and pauses while the tab is hidden.
 */
(function () {
  'use strict';

  var SIZE = 64;
  var FADE_MS = 3000; // time to fade from one colour to the next
  var HOLD_MS = 1500; // time each colour stays put
  var FPS = 12; // a slow fade needs few frames; keeps CPU and favicon churn low
  var GRAPHITE = '#3D3F44';
  // Solarized accents (https://ethanschoonover.com/solarized/), starting from the logo's red.
  var BALL_COLOURS = ['#DC322F', '#CB4B16', '#B58900', '#859900', '#2AA198', '#268BD2', '#6C71C4', '#D33682'];
  var DOTS = [[20, 80, '#CB4B16'], [32, 74, '#859900'], [22, 64, '#268BD2'], [40, 84, '#B58900'], [38, 62, '#6C71C4'], [48, 74, '#2AA198']];
  var BALL = [76, 26];

  if (!window.matchMedia || !document.createElement('canvas').getContext) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  // sRGB <-> OKLab (https://bottosson.github.io/posts/oklab/) so fades stay even in brightness.
  function toLinear(c) { return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  function toSrgb(c) { return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; }
  function hexToOklab(hex) {
    var n = parseInt(hex.slice(1), 16);
    var r = toLinear((n >> 16 & 255) / 255), g = toLinear((n >> 8 & 255) / 255), b = toLinear((n & 255) / 255);
    var l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    var m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    var s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
  }
  function oklabToCss(lab) {
    var l = Math.pow(lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2], 3);
    var m = Math.pow(lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2], 3);
    var s = Math.pow(lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2], 3);
    var rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
               -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
               -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
    return 'rgb(' + rgb.map(function (c) { return Math.round(Math.min(1, Math.max(0, toSrgb(c))) * 255); }).join(',') + ')';
  }
  var LABS = BALL_COLOURS.map(hexToOklab);

  function ballColour(ms) {
    var step = FADE_MS + HOLD_MS;
    var i = Math.floor(ms / step) % LABS.length;
    var t = Math.max(0, (ms % step) - HOLD_MS) / FADE_MS;
    t = t * t * (3 - 2 * t); // ease in and out
    var a = LABS[i], b = LABS[(i + 1) % LABS.length];
    return oklabToCss([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
  }

  var canvas = document.createElement('canvas');
  canvas.width = SIZE;
  canvas.height = SIZE;
  var ctx = canvas.getContext('2d');
  var k = SIZE / 180; // the mark is designed on a 180-unit box

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  function circle(cx, cy, r, colour) {
    ctx.beginPath();
    ctx.arc((24 + 1.32 * cx) * k, (24 + 1.32 * cy) * k, 1.32 * r * k, 0, 2 * Math.PI);
    ctx.fillStyle = colour;
    ctx.fill();
  }
  function draw(colour) {
    var line = 1; // 0.5 px at 32 px, as in favicon.svg
    ctx.clearRect(0, 0, SIZE, SIZE);
    roundRect(line / 2, line / 2, SIZE - line, SIZE - line, 40 * k);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    ctx.lineWidth = line;
    ctx.strokeStyle = GRAPHITE;
    ctx.stroke();
    DOTS.forEach(function (d) { circle(d[0], d[1], 5.5, d[2]); });
    circle(BALL[0], BALL[1], 12, colour);
  }

  // One icon link we own, placed last so browsers prefer it over the static ones.
  var link = document.createElement('link');
  link.rel = 'icon';
  link.type = 'image/png';
  link.sizes = SIZE + 'x' + SIZE;
  document.head.appendChild(link);

  var start = performance.now();
  var timer = null;
  function frame() {
    draw(ballColour(performance.now() - start));
    link.href = canvas.toDataURL('image/png');
  }
  function run() {
    if (timer === null) timer = setInterval(frame, 1000 / FPS);
  }
  function pause() {
    if (timer !== null) { clearInterval(timer); timer = null; }
  }
  document.addEventListener('visibilitychange', function () { if (document.hidden) pause(); else run(); });
  frame();
  if (!document.hidden) run();
})();
