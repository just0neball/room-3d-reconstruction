import { url } from '../api.js';
import { h, icon, button } from '../ui.js';
import { taskPanel, errorBlock, act } from './common.js';
import { api } from '../api.js';

export function renderApply(w) {
  const e = w.edit; const frames = (e.tracking?.frames || []).filter((f) => f.included);
  const done = frames.filter((f) => f.edited_url);
  const total = e.edit_progress?.total || frames.length;
  const root = h('div', { class: 'step-pane stack' });
  if (e.state === 'editing_frames') root.append(taskPanel(w, { title: `Editing frames${e.edit_progress?.total ? ` • ${e.edit_progress.done} of ${e.edit_progress.total}` : ''}`, hint: 'Each frame is edited inside its own mask using your chosen first edit as the style reference. Nothing outside the mask changes.' }));
  else {
    root.append(h('div', null, h('h3', null, 'Edited frames'), h('p', { class: 'meta' }, `${e.edit_progress?.done ?? done.length} of ${total} frames edited`)));
    root.append(errorBlock(w, () => act(w, () => api.editRun(e.id))));
    if (e.state === 'edited') root.append(h('div', { class: 'row' }, button('Continue to rebuild', { kind: 'primary', icon: 'arrow', onClick: () => w.goStep(6) })));
  }
  root.append(done.length ? h('div', { class: 'frame-grid preview' }, done.slice(-24).map((f) => h('div', { class: 'thumb' }, h('div', { class: 'thumb-img' }, h('img', { src: url(f.edited_url), alt: '', loading: 'lazy' })), h('span', { class: 'thumb-tag' }, `${(f.seconds ?? 0).toFixed(1)}s`))))
    : h('div', { class: 'empty small' }, icon('film', 24), h('p', null, e.state === 'editing_frames' ? 'Edited frames appear here as they finish.' : 'No edited frames yet.')));
  return root;
}
