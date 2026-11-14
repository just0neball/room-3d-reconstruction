import { api } from '../api.js';
import { h, icon, button, fmtDuration } from '../ui.js';
import { taskPanel, errorBlock, act } from './common.js';

export function renderRebuild(w) {
  const e = w.edit;
  if (e.state === 'rebuilding' || (w.busy && e.state === 'rebuilding')) return h('div', { class: 'step-pane narrow-pane' }, taskPanel(w, { title: 'Rebuilding the 3D scene', hint: 'Camera poses stay the same. Only the training images change.' }));
  if (e.state === 'done' && e.result?.splat_url) return doneView(w);
  const C = (w.rb ||= { mode: e.settings?.rebuild_mode || 'finetune', steps: null });
  const defaultSteps = (m) => (m === 'finetune' ? 3000 : 7000);
  C.steps ??= e.settings?.rebuild_steps || defaultSteps(C.mode);
  const steps = h('input', { class: 'input num', type: 'number', min: 500, max: 30000, step: 500, value: C.steps, 'aria-label': 'Training steps', onInput: (ev) => { C.steps = +ev.target.value || defaultSteps(C.mode); } });
  const mode = (value, title, desc, badge) => {
    const id = `rb-${value}`;
    return h('label', { class: `choice ${C.mode === value ? 'on' : ''}`, for: id },
      h('input', { type: 'radio', name: 'rbmode', id, value, checked: C.mode === value, onChange: () => { C.mode = value; C.steps = defaultSteps(value); steps.value = C.steps; w.render(true); } }),
      h('div', null, h('strong', null, title, badge ? h('span', { class: 'badge' }, badge) : null), h('p', { class: 'meta' }, desc)));
  };
  return h('div', { class: 'step-pane narrow-pane stack' },
    h('div', null, h('h3', null, 'Rebuild the 3D scene'), h('p', { class: 'meta' }, 'The edited frames replace the originals. Unedited frames stay as they were, and the original scene is kept for before/after.')),
    errorBlock(w, null),
    h('div', { class: 'stack tight', role: 'radiogroup', 'aria-label': 'Rebuild mode' },
      mode('finetune', 'Fine-tune the existing scene', 'Starts from the current Gaussians and trains for a few thousand steps.', 'faster'),
      mode('scratch', 'Retrain from scratch', 'Trains a new scene from the sparse points. Slower, and the most faithful baseline.')),
    h('label', { class: 'field inline' }, h('span', null, 'Training steps'), steps),
    h('div', { class: 'row' }, button(e.state === 'failed' ? 'Retry rebuild' : 'Rebuild scene', { kind: 'primary', icon: 'cube', onClick: () => act(w, () => api.rebuild(e.id, { mode: C.mode, steps: C.steps })) })));
}

function doneView(w) {
  const e = w.edit; const r = e.result || {}; const m = e.metrics || {};
  const stat = (label, value) => (value == null || value === '' ? null : h('div', { class: 'stat' }, h('span', { class: 'stat-v' }, value), h('span', { class: 'stat-l' }, label)));
  return h('div', { class: 'step-pane narrow-pane stack' },
    h('div', { class: 'done-head' }, h('span', { class: 'done-ico' }, icon('check', 22)), h('div', null, h('h3', null, 'Your edited room is ready'), h('p', { class: 'meta' }, `“${e.prompt || e.name}”`))),
    h('div', { class: 'stats' }, stat('rebuild mode', r.mode === 'scratch' ? 'From scratch' : r.mode === 'finetune' ? 'Fine-tune' : r.mode), stat('training steps', r.steps?.toLocaleString?.()), stat('build time', r.build_seconds != null ? fmtDuration(r.build_seconds) : null),
      stat('seconds per frame', m.edit?.sec_per_frame != null ? Number(m.edit.sec_per_frame).toFixed(1) : null), stat('peak VRAM (edit)', m.edit?.peak_vram_gb != null ? `${Number(m.edit.peak_vram_gb).toFixed(1)} GB` : null)),
    h('div', { class: 'row' }, button('View before / after', { kind: 'primary', icon: 'compare', onClick: () => w.hooks.viewResult(e) }),
      button('Rebuild again', { kind: 'ghost', icon: 'refresh', onClick: () => { w.rb = null; w.edit.state = 'edited'; w.render(true); } })));
}
