// DOM helpers, icons, toasts, dialogs and shared formatting.
export function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'for') el.htmlFor = v;
    else if (k in el && !k.includes('-') && k !== 'list' && k !== 'style') { try { el[k] = v; } catch { el.setAttribute(k, v); } }
    else el.setAttribute(k, v === true ? '' : v);
  }
  append(el, kids);
  return el;
}
export function append(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    el.append(k.nodeType ? k : document.createTextNode(String(k)));
  }
  return el;
}
export const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
export const $ = (sel, root = document) => root.querySelector(sel);

const ICONS = {
  back: 'M15 18l-6-6 6-6', plus: 'M12 5v14M5 12h14', minus: 'M5 12h14', check: 'M5 12.5l4.5 4.5L19 7', x: 'M6 6l12 12M18 6L6 18',
  upload: 'M12 16V4m0 0L8 8m4-4l4 4M4 20h16', undo: 'M9 14L4 9l5-5M4 9h10a6 6 0 010 12h-3',
  cube: 'M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3zm0 9l8-4.5M12 12v9M12 12L4 7.5',
  target: 'M12 8a4 4 0 100 8 4 4 0 000-8zM12 2v4M12 18v4M2 12h4M18 12h4',
  orbit: 'M12 10a2 2 0 100 4 2 2 0 000-4zM4 12c0-2.2 3.6-4 8-4s8 1.8 8 4-3.6 4-8 4-8-1.8-8-4zM8 4.8c1.8-1 5-.4 8 2.6s4 6.2 3 8',
  fly: 'M3 11l18-8-8 18-2-8-8-2z', camera: 'M4 8h3l2-3h6l2 3h3v11H4zM12 17a3.5 3.5 0 100-7 3.5 3.5 0 000 7z',
  layers: 'M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5', gpu: 'M4 7h16v10H4zM8 21v-4M16 21v-4M8 3v4M16 3v4',
  refresh: 'M20 11a8 8 0 10-2 6M20 4v7h-7', trash: 'M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13', film: 'M4 4h16v16H4zM8 4v16M16 4v16M4 9h4M4 15h4M16 9h4M16 15h4',
  edit: 'M4 20h4L19 9l-4-4L4 16v4zM13 7l4 4', arrow: 'M5 12h14M13 6l6 6-6 6', cursor: 'M5 3l14 7-6 2-2 6L5 3z',
  warn: 'M12 3l10 18H2L12 3zM12 10v5M12 18v.5', info: 'M12 3a9 9 0 100 18 9 9 0 000-18zM12 11v6M12 7.5v.5',
  compare: 'M12 3v18M4 6h5v12H4zM15 6h5v12h-5z', video: 'M3 6h12v12H3zM15 10l6-3v10l-6-3',
};
export function icon(name, size = 18) {
  const s = h('span', { class: 'ico', 'aria-hidden': 'true' });
  s.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="${ICONS[name] || ICONS.info}"/></svg>`;
  return s;
}
export const spinner = (cls = '') => h('span', { class: `spinner ${cls}`, role: 'status', 'aria-label': 'Loading' });

export function button(label, { kind = '', icon: ic, onClick, disabled, title, type = 'button', cls = '' } = {}) {
  return h('button', { class: `btn ${kind} ${cls}`.trim(), type, disabled, title, onClick }, ic ? icon(ic, 16) : null, label ? h('span', null, label) : null);
}

/** Segmented control. options: [{value,label}] */
export function segmented(options, value, onChange, label = '') {
  const wrap = h('div', { class: 'seg', role: 'radiogroup', 'aria-label': label });
  for (const o of options) {
    wrap.append(h('button', {
      type: 'button', role: 'radio', class: o.value === value ? 'on' : '', 'aria-checked': o.value === value ? 'true' : 'false', title: o.title,
      onClick: () => { for (const b of wrap.children) { b.classList.remove('on'); b.setAttribute('aria-checked', 'false'); }
        wrap.children[options.indexOf(o)].classList.add('on'); wrap.children[options.indexOf(o)].setAttribute('aria-checked', 'true'); onChange(o.value); },
    }, o.icon ? icon(o.icon, 15) : null, o.label));
  }
  return wrap;
}

export function progressBar(pct, { indeterminate = false, label } = {}) {
  const v = Math.max(0, Math.min(100, Math.round(pct || 0)));
  const bar = h('div', { class: `progress ${indeterminate ? 'indet' : ''}`, role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': v, 'aria-label': label || 'Progress' },
    h('div', { class: 'progress-fill', style: { width: `${indeterminate ? 40 : v}%` } }));
  return bar;
}

// ---- toasts ----
let toastRoot;
export function toast(message, kind = 'info', ms = 4500) {
  if (!toastRoot) { toastRoot = h('div', { class: 'toasts', 'aria-live': 'polite' }); document.body.append(toastRoot); }
  const t = h('div', { class: `toast ${kind}` }, icon(kind === 'error' ? 'warn' : kind === 'ok' ? 'check' : 'info', 16), h('span', null, message));
  toastRoot.append(t);
  setTimeout(() => t.remove(), ms);
  return t;
}
export const errMsg = (e) => (e && (e.detail || e.message)) || 'Something went wrong.';

// ---- dialog ----
export function openDialog({ title, body, actions = [], wide = false, onClose }) {
  const dlg = h('dialog', { class: `dialog ${wide ? 'wide' : ''}`, 'aria-label': title });
  const close = () => { try { dlg.close(); } catch { /* already closed */ } };
  dlg.append(
    h('header', { class: 'dialog-head' }, h('h2', null, title), h('button', { class: 'btn ghost icon', type: 'button', 'aria-label': 'Close', onClick: close }, icon('x', 16))),
    h('div', { class: 'dialog-body' }, body),
    actions.length ? h('footer', { class: 'dialog-foot' }, actions.map((a) => button(a.label, { kind: a.primary ? 'primary' : a.kind || '', disabled: a.disabled, onClick: () => a.onClick?.({ close, dlg }) }))) : null,
  );
  dlg.addEventListener('close', () => { dlg.remove(); onClose?.(); });
  document.body.append(dlg);
  dlg.showModal();
  return { close, dlg };
}

// ---- formatting ----
export function fmtBytes(n) {
  if (n == null) return '';
  const u = ['B', 'KB', 'MB', 'GB']; let i = 0; let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v >= 10 || i === 0 ? Math.round(v) : v.toFixed(1)} ${u[i]}`;
}
export function fmtDuration(s) {
  if (s == null) return '';
  s = Math.round(s); const m = Math.floor(s / 60); const r = s % 60;
  return m ? `${m}m ${String(r).padStart(2, '0')}s` : `${r}s`;
}
export function timeAgo(iso) {
  if (!iso) return '';
  const d = (Date.now() - new Date(iso).getTime()) / 1000;
  if (d < 45) return 'just now';
  if (d < 3600) return `${Math.round(d / 60)} min ago`;
  if (d < 86400) return `${Math.round(d / 3600)} h ago`;
  return `${Math.round(d / 86400)} d ago`;
}

export const EDIT_STATE = {
  draft: ['Draft', 'idle'], masked: ['Mask ready', 'idle'], tracking: ['Tracking object', 'busy'], tracked: ['Tracked', 'idle'],
  editing_reference: ['Editing first frame', 'busy'], reference_ready: ['Choose first edit', 'idle'], editing_frames: ['Editing frames', 'busy'],
  edited: ['Frames edited', 'idle'], rebuilding: ['Rebuilding scene', 'busy'], done: ['Done', 'ok'], failed: ['Failed', 'err'],
};
export const stateChip = (state) => { const [l, k] = EDIT_STATE[state] || [state || 'Unknown', 'idle']; return h('span', { class: `chip ${k}` }, k === 'busy' ? h('i', { class: 'pulse' }) : null, l); };
export const isActiveTask = (t) => !!t && (t.status === 'pending' || t.status === 'running');

export function projectStatus(p) {
  const t = p?.latest_task;
  if (t && (t.status === 'pending' || t.status === 'running') && t.type === 'reconstruction') return { kind: 'busy', label: t.status === 'pending' ? 'Queued' : `Reconstructing ${t.progress}%` };
  if (p?.scene_ready) return { kind: 'ok', label: 'Scene ready' };
  if (t && t.status === 'failed') return { kind: 'err', label: 'Reconstruction failed' };
  if (p?.has_video) return { kind: 'idle', label: 'Video uploaded' };
  return { kind: 'idle', label: 'No video yet' };
}
