import { api, url } from '../api.js';
import { h, icon, button } from '../ui.js';
import { taskPanel, errorBlock, act, frameUrl } from './common.js';

export function renderDescribe(w) {
  const e = w.edit;
  if (w.busy && e.state === 'tracking') {
    return h('div', { class: 'step-pane narrow-pane stack' }, taskPanel(w, { title: 'Finding the object in every frame', hint: 'Your mask is lifted into 3D with the camera poses, then each frame is segmented. This can take a few minutes.' }));
  }
  const prompt = h('textarea', { class: 'input prompt', rows: 3, placeholder: 'make the sofa dark green leather', 'aria-label': 'Edit prompt', maxLength: 400, value: e.prompt || '' });
  const method = h('select', { class: 'select', 'aria-label': 'Tracking method' },
    h('option', { value: 'auto' }, 'Automatic (recommended)'), h('option', { value: 'lift3d' }, 'Lift to 3D and reproject'), h('option', { value: 'video' }, 'Video mask propagation'), h('option', { value: 'neighbor' }, 'Neighbor-mask search'));
  const go = button(e.tracking ? 'Track again' : 'Find object in all frames', { kind: 'primary', icon: 'target', onClick: async () => {
    const text = prompt.value.trim();
    if (!text) { prompt.focus(); prompt.classList.add('invalid'); return; }
    go.disabled = true;
    await act(w, async () => { await api.patchEdit(e.id, { prompt: text }); await api.track(e.id, { method: method.value }); });
    go.disabled = false;
  } });
  prompt.addEventListener('input', () => prompt.classList.remove('invalid'));
  const refUrl = e.ref_frame ? frameUrl(w, e.ref_frame) : null;
  return h('div', { class: 'step-pane two-col' },
    h('div', { class: 'ref-preview' }, refUrl ? h('div', { class: 'ref-img' }, h('img', { src: url(refUrl), alt: 'Selected frame' }), e.ref_overlay_url ? h('img', { class: 'ov', src: url(e.ref_overlay_url), alt: '' }) : null) : h('div', { class: 'empty small' }, icon('target', 24)),
      button('Change selection', { kind: 'ghost sm', icon: 'back', onClick: () => w.goStep(1) })),
    h('div', { class: 'stack' },
      h('div', null, h('h3', null, 'What should it become?'), h('p', { class: 'meta' }, 'Describe the change in plain words. The same description is used for every frame.')),
      errorBlock(w, null),
      h('label', { class: 'field' }, h('span', null, 'Edit prompt'), prompt),
      e.tracking ? h('p', { class: 'hint' }, 'Tracking again discards the current frame selection and any edits made from it.') : null,
      h('details', { class: 'advanced' }, h('summary', null, 'Advanced: tracking method'), h('div', { class: 'stack tight' }, method,
        h('p', { class: 'hint' }, 'Automatic lifts your mask to 3D using the camera poses, then cross-checks with video propagation. Pick another method to compare results.'))),
      h('div', { class: 'row' }, go)));
}
