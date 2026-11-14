import { api, url } from '../api.js';
import { h, icon, button, segmented, toast, errMsg, clear } from '../ui.js';
import { MaskCanvas } from './maskcanvas.js';

export function renderPick(w) {
  const e = w.edit;
  const P = (w.pick ||= { tab: 'frames', k: 1, frame: e.ref_frame || null, points: (e.points || []).slice(), seq: 0, timer: null, mask: e.ref_mask_url ? { mask_url: e.ref_mask_url } : null });
  let pending = Promise.resolve();
  const root = h('div', { class: 'pick' });
  w.view = 1; // stay on this step while the user is working here; leaving it is explicit (Use this mask)

  const tabs = segmented([{ value: 'frames', label: 'Pick a frame', icon: 'film', title: 'Recommended: click on a real photo' }, { value: 'three', label: 'Click in 3D', icon: 'cube', title: 'Click the object in the 3D scene' }],
    P.tab, (v) => { P.tab = v; draw(); }, 'How to pick the object');
  const body = h('div', { class: 'pick-body' });
  root.append(h('div', { class: 'pick-top' }, h('div', null, h('h3', null, 'Which object do you want to change?'),
    h('p', { class: 'meta' }, 'Click it on a frame. Add points until the pink outline covers only that object.')), tabs), body);

  function draw() {
    clear(body);
    if (P.tab === 'three') return body.append(threeTab());
    return body.append(P.frame ? canvasView() : gridView());
  }

  function gridView() {
    const fps = w.framesInfo?.fps || 1; const all = w.framesInfo?.frames || [];
    const frames = all.filter((_, i) => i % P.k === 0);
    const label = h('span', { class: 'meta' });
    const grid = h('div', { class: 'frame-grid' });
    const fill = () => {
      const fr = all.filter((_, i) => i % P.k === 0);
      label.textContent = `${+(fps / P.k).toFixed(2)} fps • ${fr.length} of ${all.length} frames`;
      clear(grid);
      fr.forEach((f) => grid.append(h('button', { class: 'thumb', type: 'button', title: `${f.name} • ${f.seconds?.toFixed?.(1) ?? ''}s`, onClick: () => { P.frame = f.name; P.points = []; P.mask = null; draw(); } },
        h('img', { src: url(f.thumb_url), alt: `Frame at ${f.seconds?.toFixed?.(1) ?? f.index}s`, loading: 'lazy' }), h('span', { class: 'thumb-tag' }, `${(f.seconds ?? 0).toFixed(1)}s`))));
    };
    fill();
    if (!all.length) return h('div', { class: 'empty small' }, icon('film', 24), h('p', null, 'No training frames were found for this scene.'));
    return h('div', { class: 'stack' }, h('div', { class: 'row between' }, segmented([{ value: 1, label: 'Every frame' }, { value: 2, label: 'Every 2nd' }, { value: 4, label: 'Every 4th' }], P.k, (v) => { P.k = v; fill(); }, 'Frame sampling'), label),
      h('p', { class: 'hint' }, 'Choose a frame where the object is large and clearly visible.'), grid);
  }

  function canvasView() {
    const f = w.frameMap.get(P.frame);
    const info = h('p', { class: 'meta mask-info', 'aria-live': 'polite' });
    const next = button('Use this mask', { kind: 'primary', icon: 'arrow', disabled: !P.mask, onClick: async () => {
      clearTimeout(P.timer); await pending;
      if (P.points.length && !P.mask) await run();
      if (!P.mask) return;
      try { await w.refresh(); } catch { /* ignore */ }
      w.view = 2; w.render(true);
    } });
    const mc = new MaskCanvas({ onChange: (pts) => { P.points = pts.map((p) => ({ ...p })); debounce(); } });
    P.mc = mc;
    const updateInfo = () => {
      const m = P.mask;
      info.textContent = m ? `Object covers ${m.area != null ? Math.round(m.area * 100) : '?'}% of the frame${m.score != null ? ` • confidence ${Math.round(m.score * 100)}%` : ''}` : (P.points.length ? 'Segmenting…' : 'No object selected yet. Click on it.');
      next.disabled = !P.mask;
    };
    const debounce = () => { clearTimeout(P.timer); P.timer = setTimeout(() => { pending = run(); }, 220); updateInfo(); };
    const run = async () => {
      if (!P.points.length) { P.seq++; P.mask = null; await mc.setMask(null); mc.setBusy(false); updateInfo(); return; }
      const seq = ++P.seq; mc.setBusy(true);
      try {
        const r = await api.segment(e.id, { frame: P.frame, points: P.points, set_reference: true });
        if (seq !== P.seq) return;
        P.mask = r; await mc.setMask(r.mask_url);
        Object.assign(w.edit, { ref_frame: P.frame, points: P.points.slice(), ref_mask_url: r.mask_url, ref_overlay_url: r.overlay_url });
      } catch (err) { if (seq === P.seq) toast(errMsg(err), 'error', 7000); }
      finally { if (seq === P.seq) { mc.setBusy(false); updateInfo(); } }
    };
    P.runSegment = run;
    mc.setImage(f?.url, w.framesInfo?.image_size).then(async () => { mc.setPoints(P.points); if (P.mask?.mask_url) await mc.setMask(P.mask.mask_url); if (P.points.length && !P.mask) debounce(); });
    const mode = segmented([{ value: 'pos', label: 'Add to object', icon: 'plus' }, { value: 'neg', label: 'Exclude', icon: 'minus' }], 'pos', (v) => mc.setMode(v), 'Click mode');
    updateInfo();
    return h('div', { class: 'pick-canvas' },
      h('div', { class: 'canvas-wrap' }, mc.el),
      h('aside', { class: 'tool-col stack' },
        h('div', { class: 'stack tight' }, h('strong', null, f?.name || P.frame), h('p', { class: 'meta' }, f ? `${(f.seconds ?? 0).toFixed(1)}s into the video` : '')),
        mode,
        h('div', { class: 'hint-list' }, h('p', null, h('span', { class: 'kbd' }, 'Click'), ' adds a point on the object.'), h('p', null, h('span', { class: 'kbd' }, 'Right-click'), ' or ', h('span', { class: 'kbd' }, 'Alt'), '+click excludes an area.')),
        h('div', { class: 'row' }, button('Undo', { icon: 'undo', kind: 'sm', onClick: () => { mc.undo(); } }), button('Clear', { kind: 'ghost sm', onClick: () => mc.clear() })),
        info,
        h('div', { class: 'row' }, button('Other frame', { kind: 'ghost sm', onClick: () => { P.frame = null; P.points = []; P.mask = null; draw(); } })),
        next));
  }

  function threeTab() {
    return h('div', { class: 'three-tab stack' },
      h('div', { class: 'callout' }, icon('info', 18), h('p', null, 'Harder than a photo: you aim in the 3D view and the nearest training frame that sees that spot is opened for you to refine.')),
      button('Click an object in the 3D view', { kind: 'primary', icon: 'cube', onClick: async () => {
        const p = await w.hooks.pick3d(); if (!p) return;
        try {
          const r = await api.pick3d(e.id, p);
          P.frame = r.frame; P.points = [{ x: Math.round(r.x), y: Math.round(r.y), label: 1 }]; P.mask = null; P.tab = 'frames';
          tabs.querySelectorAll('button').forEach((b, i) => { b.classList.toggle('on', i === 0); b.setAttribute('aria-checked', i === 0 ? 'true' : 'false'); });
          draw();
        } catch (err) { toast(errMsg(err), 'error', 7000); }
      } }));
  }
  draw();
  w._keep = true;
  return root;
}
