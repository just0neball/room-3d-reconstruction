import { api, url } from '../api.js';
import { h, icon, button, segmented, openDialog, toast, errMsg, clear } from '../ui.js';
import { MaskCanvas } from './maskcanvas.js';
import { taskPanel, errorBlock, act, frameUrl, pct } from './common.js';

export function renderReview(w) {
  const e = w.edit; const T = e.tracking;
  if (w.busy && e.state === 'tracking') return h('div', { class: 'step-pane narrow-pane' }, taskPanel(w, { title: 'Segmenting frames', hint: 'Marked frames are segmented one by one.' }));
  if (!T) return h('div', { class: 'step-pane narrow-pane stack' }, errorBlock(w, null), h('div', { class: 'empty small' }, icon('film', 24), h('p', null, 'Nothing has been tracked yet.'), button('Go to step 2', { onClick: () => w.goStep(2) })));

  const R = (w.review ||= { mode: 'auto', filter: 'all', marked: null, last: null });
  const frames = T.frames || [];
  if (!R.marked) R.marked = new Set(frames.filter((f) => f.included).map((f) => f.name));
  const included = () => frames.filter((f) => f.included).length;
  const missing = () => frames.filter((f) => f.included && !f.mask_url).length;

  const summary = h('p', { class: 'summary', 'aria-live': 'polite' });
  const grid = h('div', { class: 'frame-grid review' });
  const actions = h('div', { class: 'row' });
  const root = h('div', { class: 'step-pane stack' });

  const updateSummary = () => {
    summary.textContent = R.mode === 'manual' ? `${R.marked.size} of ${frames.length} frames marked as containing the object`
      : `${included()} of ${frames.length} frames include the object${T.method ? ` • tracked with ${T.method}` : ''}`;
  };

  let queue = Promise.resolve();
  const toggleIncluded = (f, on) => {
    f.included = on; updateSummary(); renderActions();
    queue = queue.then(() => api.patchFrames(e.id, on ? { include: [f.name] } : { exclude: [f.name] })).catch((err) => { toast(errMsg(err), 'error'); });
  };

  function shown() {
    return frames.filter((f) => R.filter === 'all' || (R.filter === 'included' ? f.included : !f.included));
  }
  function fillGrid() {
    clear(grid);
    const list = shown();
    if (!list.length) grid.append(h('div', { class: 'empty small wide' }, h('p', null, 'No frames match this filter.')));
    list.forEach((f, idx) => {
      const manual = R.mode === 'manual';
      const on = manual ? R.marked.has(f.name) : f.included;
      const img = (!manual && f.overlay_url) || f.thumb_url;
      const cb = h('input', { type: 'checkbox', checked: on, 'aria-label': `${manual ? 'Mark' : 'Include'} frame at ${(f.seconds ?? 0).toFixed(1)}s`,
        onClick: (ev) => {
          if (manual) {
            const checked = ev.target.checked;
            if (ev.shiftKey && R.last != null) { const [a, b] = [Math.min(R.last, idx), Math.max(R.last, idx)]; list.slice(a, b + 1).forEach((x) => (checked ? R.marked.add(x.name) : R.marked.delete(x.name))); fillGrid(); }
            else (checked ? R.marked.add(f.name) : R.marked.delete(f.name));
            R.last = idx; updateSummary(); renderActions();
          } else { toggleIncluded(f, ev.target.checked); card.classList.toggle('off', !ev.target.checked); }
        } });
      const card = h('div', { class: `thumb review-card ${on ? 'sel' : ''} ${!manual && !on ? 'off' : ''}` },
        h('button', { class: 'thumb-img', type: 'button', title: manual ? 'Toggle' : 'Click to refine this frame’s mask', onClick: () => (manual ? cb.click() : refine(f)) }, h('img', { src: url(img), alt: '', loading: 'lazy' })),
        h('label', { class: 'thumb-check' }, cb, h('span', { class: 'visually-hidden' }, 'Include')),
        h('span', { class: 'thumb-tag' }, `${(f.seconds ?? 0).toFixed(1)}s`),
        !manual && f.included && f.source && f.source !== 'none' ? h('span', { class: `thumb-badge ${f.source}` }, f.source === 'reference' ? 'reference' : f.source === 'manual' ? 'manual' : f.score != null ? `${Math.round(f.score * 100)}%` : 'auto') : null,
        !manual && f.included && !f.mask_url ? h('span', { class: 'thumb-warn', title: 'Included but has no mask' }, icon('warn', 13)) : null);
      grid.append(card);
    });
  }

  function refine(f) {
    const src = frameUrl(w, f.name); if (!src) return;
    let seq = 0; let points = []; let timer;
    const mc = new MaskCanvas({ onChange: (pts) => { points = pts.map((p) => ({ ...p })); clearTimeout(timer); timer = setTimeout(run, 220); } });
    const run = async () => {
      if (!points.length) { mc.setMask(f.mask_url || null); return; }
      const s = ++seq; mc.setBusy(true);
      try { const r = await api.segment(e.id, { frame: f.name, points, set_reference: false }); if (s !== seq) return; await mc.setMask(r.mask_url); f.mask_url = r.mask_url; f.overlay_url = r.overlay_url || r.mask_url; f.source = 'manual'; f.included = true; f.area = r.area ?? f.area; }
      catch (err) { toast(errMsg(err), 'error', 7000); } finally { if (s === seq) mc.setBusy(false); }
    };
    mc.setImage(src, w.framesInfo?.image_size).then(() => f.mask_url && mc.setMask(f.mask_url));
    openDialog({ title: `Refine mask • ${(f.seconds ?? 0).toFixed(1)}s`, wide: true,
      body: h('div', { class: 'stack' }, mc.el, h('p', { class: 'hint' }, 'Click to add the object, right-click or Alt+click to exclude. The first click replaces this frame’s mask.'),
        h('div', { class: 'row' }, button('Undo', { icon: 'undo', kind: 'sm', onClick: () => mc.undo() }), button('Clear points', { kind: 'ghost sm', onClick: () => { mc.clear(); mc.setMask(f.mask_url || null); } }))),
      actions: [{ label: 'Done', primary: true, onClick: async ({ close }) => { close(); await queue; try { await w.refresh(); } catch { /* ignore */ } } }], onClose: () => clearTimeout(timer) });
  }

  function renderActions() {
    clear(actions);
    if (R.mode === 'manual') {
      actions.append(
        button(`Segment ${R.marked.size} marked frame${R.marked.size === 1 ? '' : 's'}`, { kind: 'primary', icon: 'target', disabled: !R.marked.size, onClick: () => act(w, () => { const fr = [...R.marked]; R.mode = 'auto'; R.marked = null; return api.track(e.id, { method: 'manual', frames: fr }); }) }),
        button('Select all shown', { kind: 'ghost sm', onClick: () => { shown().forEach((f) => R.marked.add(f.name)); fillGrid(); updateSummary(); renderActions(); } }),
        button('Clear', { kind: 'ghost sm', onClick: () => { R.marked.clear(); fillGrid(); updateSummary(); renderActions(); } }));
    } else {
      const miss = missing();
      actions.append(
        button('Continue to first edit', { kind: 'primary', icon: 'arrow', disabled: included() < 1, onClick: async () => { await queue; w.goStep(4); } }),
        miss ? button(`Segment ${miss} frame${miss === 1 ? '' : 's'} without a mask`, { icon: 'target', onClick: () => act(w, () => { R.marked = null; return api.track(e.id, { method: 'manual', frames: frames.filter((f) => f.included && !f.mask_url).map((f) => f.name) }); }) }) : null);
    }
  }

  const modeSeg = segmented([{ value: 'auto', label: 'Review automatic result', icon: 'target' }, { value: 'manual', label: 'Mark frames manually', icon: 'cursor', title: 'Fallback: tick every frame that shows the object' }], R.mode,
    (v) => { R.mode = v; if (v === 'manual') R.marked = new Set(frames.filter((f) => f.included).map((f) => f.name)); R.last = null; paint(); }, 'Review mode');
  const filterSeg = segmented([{ value: 'all', label: 'All' }, { value: 'included', label: 'Included' }, { value: 'missing', label: 'Missing' }], R.filter, (v) => { R.filter = v; R.last = null; fillGrid(); }, 'Filter frames');
  const note = h('p', { class: 'hint' });

  function paint() {
    note.textContent = R.mode === 'manual'
      ? 'Manual path: tick every frame where the object clearly appears (Shift+click selects a range). The segmentation model then runs on exactly those frames.'
      : 'Automatic path: untick frames where the mask is wrong, or click a frame to fix its mask. Frames the tracker missed can be added in “Mark frames manually”.';
    updateSummary(); fillGrid(); renderActions();
  }

  root.append(
    h('div', { class: 'row between wrap' }, h('div', null, h('h3', null, 'Check where the object was found'), summary), modeSeg),
    errorBlock(w, null), note,
    h('div', { class: 'row between wrap' }, filterSeg, actions),
    grid);
  paint();
  w._keep = true;
  return root;
}
