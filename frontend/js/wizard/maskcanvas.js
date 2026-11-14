import { h, icon, spinner } from '../ui.js';
import { url } from '../api.js';

const MASK_RGB = [255, 79, 154];

function loadImage(src) {
  return new Promise((res, rej) => { const i = new Image(); i.crossOrigin = 'anonymous'; i.onload = () => res(i); i.onerror = () => rej(new Error(`Could not load image`)); i.src = src; });
}

/**
 * Frame canvas with click-to-prompt. Left click = positive point, right click / Alt+click / "Remove" mode = negative.
 * The mask is drawn as a tinted fill plus a crisp outline. Coordinates are image pixels.
 */
export class MaskCanvas {
  constructor({ onChange, readOnly = false } = {}) {
    this.onChange = onChange; this.readOnly = readOnly;
    this.points = []; this.mode = 'pos'; this.img = null; this.fill = null; this.edge = null;
    this.canvas = h('canvas', { class: 'mc-canvas', tabindex: 0, 'aria-label': 'Frame. Click the object; right click or Alt+click to exclude.' });
    this.busy = h('div', { class: 'mc-busy hidden', 'aria-live': 'polite' }, spinner(), h('span', null, 'Segmenting…'));
    this.err = h('div', { class: 'mc-err hidden' });
    this.el = h('div', { class: `mc ${readOnly ? 'ro' : ''}` }, this.canvas, this.busy, this.err);
    this.ctx = this.canvas.getContext('2d');
    if (!readOnly) {
      this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
      this.canvas.addEventListener('pointerdown', (e) => this._click(e));
    }
  }
  async setImage(src, size) {
    try {
      this.err.classList.add('hidden');
      this.img = await loadImage(url(src));
      this.canvas.width = size?.width || this.img.naturalWidth; this.canvas.height = size?.height || this.img.naturalHeight;
      this.draw();
    } catch { this.err.textContent = 'The frame could not be loaded.'; this.err.classList.remove('hidden'); }
  }
  async setMask(src) {
    if (!src) { this.fill = this.edge = null; this.draw(); return; }
    try {
      const m = await loadImage(url(src));
      const w = this.canvas.width; const hh = this.canvas.height;
      const tmp = document.createElement('canvas'); tmp.width = w; tmp.height = hh;
      const tc = tmp.getContext('2d', { willReadFrequently: true }); tc.drawImage(m, 0, 0, w, hh);
      const src32 = tc.getImageData(0, 0, w, hh).data;
      const on = new Uint8Array(w * hh);
      for (let i = 0; i < on.length; i++) on[i] = src32[i * 4 + 3] > 127 && (src32[i * 4] + src32[i * 4 + 1] + src32[i * 4 + 2]) / 3 > 127 ? 1 : 0;
      const fill = new ImageData(w, hh); const edge = new ImageData(w, hh);
      for (let y = 0; y < hh; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x; if (!on[i]) continue;
        const o = i * 4; fill.data[o] = MASK_RGB[0]; fill.data[o + 1] = MASK_RGB[1]; fill.data[o + 2] = MASK_RGB[2]; fill.data[o + 3] = 92;
        if (x === 0 || y === 0 || x === w - 1 || y === hh - 1 || !on[i - 1] || !on[i + 1] || !on[i - w] || !on[i + w]) { edge.data[o] = 255; edge.data[o + 1] = 120; edge.data[o + 2] = 178; edge.data[o + 3] = 255; }
      }
      this.fill = await createImageBitmap(fill); this.edge = await createImageBitmap(edge);
    } catch { this.fill = this.edge = null; }
    this.draw();
  }
  setPoints(points) { this.points = points.map((p) => ({ ...p })); this.draw(); }
  setMode(m) { this.mode = m; }
  setBusy(b) { this.busy.classList.toggle('hidden', !b); }
  undo() { this.points.pop(); this.draw(); this.onChange?.(this.points); }
  clear() { this.points = []; this.fill = this.edge = null; this.draw(); this.onChange?.(this.points); }
  _click(e) {
    if (e.button !== 0 && e.button !== 2) return;
    e.preventDefault();
    const r = this.canvas.getBoundingClientRect();
    const x = Math.round(((e.clientX - r.left) * this.canvas.width) / r.width); const y = Math.round(((e.clientY - r.top) * this.canvas.height) / r.height);
    const neg = e.button === 2 || e.altKey || this.mode === 'neg';
    this.points.push({ x, y, label: neg ? 0 : 1 }); this.draw(); this.onChange?.(this.points);
  }
  draw() {
    const { ctx, canvas } = this; ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (this.img) ctx.drawImage(this.img, 0, 0, canvas.width, canvas.height);
    if (this.fill) ctx.drawImage(this.fill, 0, 0);
    if (this.edge) ctx.drawImage(this.edge, 0, 0);
    const r = Math.max(6, canvas.width / 90);
    for (const p of this.points) {
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fillStyle = p.label ? '#3ddc97' : '#ff5a5a'; ctx.fill();
      ctx.lineWidth = Math.max(2, r / 3); ctx.strokeStyle = '#0b0e14'; ctx.stroke();
      ctx.beginPath(); ctx.lineWidth = Math.max(2, r / 3.2); ctx.strokeStyle = '#0b0e14';
      ctx.moveTo(p.x - r * 0.5, p.y); ctx.lineTo(p.x + r * 0.5, p.y); if (p.label) { ctx.moveTo(p.x, p.y - r * 0.5); ctx.lineTo(p.x, p.y + r * 0.5); } ctx.stroke();
    }
  }
}
